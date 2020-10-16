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
