import pytorch_lightning as pl
import torch
import torch.nn as nn


class Autoencoder(pl.LightningModule):
    def __init__(self, encoder, bottleneck, decoder, lr=0.01, noise_ratio=None):
        super().__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        self.lr = lr
        self.noise_ratio = noise_ratio or 0.
        self.criterion_recon = nn.BCELoss(reduction='none')
        self.add_noise = AddNoise(self.noise_ratio)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs):
        inputs = self.add_noise(inputs)
        encoded = self.encoder(inputs)
        latent_code, _ = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        return decoded

    def training_step(self, inputs, batch_idx):
        inputs, _ = inputs
        loss, bottleneck_loss, recon_loss = self._get_losses(inputs)

        self.log('train/recon', recon_loss)
        self.log('train/bottleneck', bottleneck_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, inputs, batch_idx):
        self._evaluate(inputs, batch_idx, mode='val')

    def test_step(self, inputs, batch_idx):
        self._evaluate(inputs, batch_idx, mode='test')

    def _evaluate(self, inputs, batch_idx, mode):
        inputs, _, = inputs
        if mode == 'val' and batch_idx == 0:
            self._log_generate_images(inputs, mode)

        loss, bottleneck_loss, recon_loss = self._get_losses(inputs)

        self.log(f'{mode}/recon', recon_loss)
        self.log(f'{mode}/bottleneck', bottleneck_loss)
        self.log(f'{mode}/loss', loss)

    def _log_generate_images(self, inputs, mode):
        outputs = self(inputs)
        comparison = torch.cat([inputs, outputs], dim=2)
        self.logger.experiment.add_images(f'{mode}/reconstructions', comparison, self.global_step)

    def _get_losses(self, inputs):
        noisy_inputs = self.add_noise(inputs)
        encoded = self.encoder(noisy_inputs)
        latent_code, bottleneck_loss = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        recon_loss = self.criterion_recon(decoded, inputs).mean(0).sum()  # batch mean of binary cross entropy
        loss = recon_loss + bottleneck_loss

        return loss, bottleneck_loss, recon_loss


class AddNoise(nn.Module):
    def __init__(self, noise_ratio):
        super().__init__()
        self.noise_ratio = noise_ratio

    def forward(self, img):
        if self.training and self.noise_ratio > 0:
            img = img + torch.randn_like(img) * self.noise_ratio
            img = torch.clamp(img, min=0., max=1.)

        return img
