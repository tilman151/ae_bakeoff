import torch
import torch.nn as nn
import pytorch_lightning as pl


class Autoencoder(pl.LightningModule):
    def __init__(self, encoder, bottleneck, decoder, lr=0.01):
        super().__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        self.lr = lr
        self.criterion_recon = nn.MSELoss(reduction='mean')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        latent_code, _ = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        return decoded

    def training_step(self, inputs, batch_idx):
        inputs, _ = inputs
        encoded = self.encoder(inputs)
        latent_code, bottleneck_loss = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        recon_loss = self.criterion_recon(inputs, decoded)
        loss = recon_loss + bottleneck_loss

        self.log('train/recon', recon_loss)
        self.log('train/bottleneck', bottleneck_loss)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, inputs, batch_idx):
        inputs, _, = inputs
        if batch_idx == 0:
            outputs = self(inputs)
            comparison = torch.cat([inputs, outputs], dim=2)
            self.logger.experiment.add_images('val/reconstructions', comparison, self.global_step)

        encoded = self.encoder(inputs)
        latent_code, bottleneck_loss = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        recon_loss = self.criterion_recon(inputs, decoded)
        loss = recon_loss + bottleneck_loss

        self.log('val/recon', recon_loss)
        self.log('val/bottleneck', bottleneck_loss)
        self.log('val/loss', loss)

    # def test_step(self, inputs):
    #     pass
