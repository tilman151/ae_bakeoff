import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def forward(self, encoded):
        """Calculate latent code and loss."""
        raise NotImplementedError

    def _loss(self, *args, **kwargs):
        """Calculate the loss of the bottleneck."""
        raise NotImplementedError


class IdentityBottleneck(Bottleneck):
    def forward(self, encoded):
        return encoded, self._loss()

    def _loss(self):
        return 0


class VariationalBottleneck(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()

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


class SparseBottleneck(Bottleneck):
    def __init__(self, sparsity, beta=1.):
        super().__init__()

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


class VectorQuantizedBottleneck(Bottleneck):
    def __init__(self, latent_dim, num_categories=512, beta=1.):
        super(VectorQuantizedBottleneck, self).__init__()

        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.beta = beta

        self.embeddings = self._build_embeddings()
        self.quantize = Quantize().apply
        self.sum_squared_error = nn.MSELoss(reduction='sum')

    def _build_embeddings(self):
        embeddings = nn.Parameter(torch.randn(1, self.latent_dim, self.num_categories))

        return embeddings

    def forward(self, encoded):
        latent_code, selected_idx = self.quantize(encoded, self.embeddings)
        loss = self._loss(encoded, latent_code)

        return latent_code, loss

    def _loss(self, encoded, latent_code):
        vq_loss = self.sum_squared_error(latent_code, encoded.detach())
        commitment_loss = self.sum_squared_error(encoded, latent_code.detach())
        loss = vq_loss + self.beta * commitment_loss

        return loss


class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, embeddings):
        _, latent_dim, num_categories = embeddings.shape
        dist = (embeddings - inputs.unsqueeze(-1)) ** 2
        dist_idx = torch.argmin(dist, dim=-1)
        offsets = torch.arange(0, latent_dim) * num_categories
        dist_idx_flat = dist_idx + offsets.repeat(inputs.shape[0], 1)
        latent_code = torch.take(embeddings.squeeze(0), dist_idx_flat)

        return latent_code, dist_idx

    @staticmethod
    def backward(ctx, grad_output, grad_embedding):
        return grad_output, None
