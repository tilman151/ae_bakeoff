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
        self.sum_squared_error = nn.MSELoss(reduction='none')
        self._straight_through_estimation = StraightThroughEstimator.apply

    def _build_embeddings(self):
        embeddings = nn.Parameter(torch.empty(1, self.num_categories))
        embeddings.data.uniform_(-1 / self.num_categories, 1 / self.num_categories)

        return embeddings

    def forward(self, encoded):
        latent_code = self._quantize(encoded)
        loss = self._loss(encoded, latent_code)
        latent_code = self._straight_through_estimation(encoded, latent_code)

        return latent_code, loss

    def _quantize(self, inputs):
        flat_inputs = inputs.view(-1, 1)
        dist = (self.embeddings - flat_inputs) ** 2
        dist_idx = torch.argmin(dist, dim=-1).unsqueeze(1)
        idx_mask = torch.zeros(flat_inputs.shape[0], self.num_categories, device=inputs.device)
        idx_mask.scatter_(1, dist_idx, 1)
        latent_code = torch.matmul(idx_mask, self.embeddings.T)
        latent_code = latent_code.view(inputs.shape)

        return latent_code

    def _loss(self, encoded, latent_code):
        vq_loss = self.sum_squared_error(latent_code, encoded.detach())
        commitment_loss = self.sum_squared_error(encoded, latent_code.detach())
        loss = vq_loss + self.beta * commitment_loss
        loss = loss.sum(1).mean(0)

        return loss


class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, encoding, quantized):
        return quantized

    @staticmethod
    def backward(ctx, grad_quantized):
        return grad_quantized, None


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = nn.functional.mse_loss(quantized.detach(), inputs, reduction='sum')
        q_latent_loss = nn.functional.mse_loss(quantized, inputs.detach(), reduction='sum')
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        loss /= input_shape[0]

        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss