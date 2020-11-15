import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, latent_dim):
        super(Bottleneck, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, encoded):
        """Calculate latent code and loss."""
        raise NotImplementedError

    def _loss(self, *args, **kwargs):
        """Calculate the loss of the bottleneck."""
        raise NotImplementedError

    def sample(self, n):
        """Sample n data points from the latent space."""
        raise NotImplementedError


class IdentityBottleneck(Bottleneck):
    def forward(self, encoded):
        return encoded, self._loss()

    def _loss(self):
        return 0

    def sample(self, n):
        return None


class VariationalBottleneck(Bottleneck):
    def __init__(self, latent_dim, beta=1.):
        super().__init__(latent_dim)

        self.beta = beta

    def forward(self, encoded):
        latent_dim = encoded.shape[1] // 2
        mu, log_sigma = torch.split(encoded, latent_dim, dim=1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu
        kl_div = self._loss(mu, log_sigma)

        return latent_code, kl_div

    def _loss(self, mu, log_sigma):
        kl_div = 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma, dim=1)
        kl_div = kl_div.mean()  # Account for batch size
        kl_div *= self.beta  # trade off

        return kl_div

    def sample(self, n):
        return torch.randn(n, self.latent_dim)


class SparseBottleneck(Bottleneck):
    def __init__(self, latent_dim, sparsity, beta=1.):
        super().__init__(latent_dim)

        self.sparsity = sparsity
        self.beta = beta

    def forward(self, encoded):
        latent_code = torch.sigmoid(encoded)
        sparsity_loss = self._loss(latent_code)

        return latent_code, sparsity_loss

    def _loss(self, latent_code):
        average_activation = torch.mean(latent_code, dim=0)  # mean over batch
        kl_div = (self.sparsity * torch.log(self.sparsity / average_activation) +
                  (1 - self.sparsity) * torch.log((1 - self.sparsity) / (1 - average_activation)))
        kl_div = torch.sum(kl_div)
        kl_div *= self.beta  # trade off

        return kl_div

    def sample(self, n):
        return None


class VectorQuantizedBottleneck(Bottleneck):
    def __init__(self, latent_dim, num_categories=512, beta=1.):
        super(VectorQuantizedBottleneck, self).__init__(latent_dim)
        self.num_categories = num_categories
        self.beta = beta

        self.embeddings = self._build_embeddings()
        self.sum_squared_error = nn.MSELoss(reduction='none')
        self._straight_through_estimation = StraightThroughEstimator.apply

    def _build_embeddings(self):
        embeddings = nn.Parameter(torch.empty(1, self.latent_dim, self.num_categories))
        embeddings.data.uniform_(-1 / self.num_categories, 1 / self.num_categories)

        return embeddings

    def forward(self, encoded):
        latent_code = self._quantize(encoded)
        loss = self._loss(encoded, latent_code)
        latent_code = self._straight_through_estimation(encoded, latent_code)

        return latent_code, loss

    def _quantize(self, inputs):
        dist = (self.embeddings - inputs.unsqueeze(-1)) ** 2
        dist_idx = torch.argmin(dist, dim=-1)
        latent_code = self._take_from_embedding(dist_idx)

        return latent_code

    def _loss(self, encoded, latent_code):
        vq_loss = self.sum_squared_error(latent_code, encoded.detach())
        commitment_loss = self.sum_squared_error(encoded, latent_code.detach())
        loss = vq_loss + self.beta * commitment_loss
        loss = loss.sum(1).mean(0)

        return loss

    def sample(self, n):
        sampled_idx = [torch.randint(self.num_categories, size=(self.latent_dim,)) for _ in range(n)]
        sampled_idx = torch.stack(sampled_idx)
        samples = self._take_from_embedding(sampled_idx)

        return samples

    def _take_from_embedding(self, idx):
        offsets = torch.arange(0, self.latent_dim) * self.num_categories
        dist_idx_flat = idx + offsets.repeat(idx.shape[0], 1)
        latent_code = torch.take(self.embeddings.squeeze(0), dist_idx_flat)

        return latent_code


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, encoding, quantized):
        return quantized

    @staticmethod
    def backward(ctx, grad_quantized):
        return grad_quantized, None
