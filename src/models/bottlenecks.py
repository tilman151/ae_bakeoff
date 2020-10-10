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
    def forward(self, encoded):
        latent_dim = encoded.shape[1] // 2
        mu, log_sigma = torch.split(encoded, latent_dim, dim=1)
        noise = torch.randn_like(mu)
        latent_code = noise * log_sigma.exp() + mu
        loss = self._loss(mu, log_sigma)

        return latent_code, loss

    def _loss(self, mu, log_sigma):
        return 0.5 * torch.sum((2 * log_sigma).exp() + mu ** 2 - 1 - 2 * log_sigma)
