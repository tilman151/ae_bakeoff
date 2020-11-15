import torch
import torch.nn as nn
import pytorch_lightning as pl

import utils


class Autoencoder(pl.LightningModule):
    def __init__(self, encoder, bottleneck, decoder, lr=0.01):
        super().__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        self.lr = lr
        self.criterion_recon = nn.BCELoss(reduction='none')

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, inputs):
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
        if batch_idx == 0:
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
        encoded = self.encoder(inputs)
        latent_code, bottleneck_loss = self.bottleneck(encoded)
        decoded = self.decoder(latent_code)

        recon_loss = self.criterion_recon(decoded, inputs).mean(0).sum()  # batch mean of L2 norm
        loss = recon_loss + bottleneck_loss

        return loss, bottleneck_loss, recon_loss


class Classifier(pl.LightningModule):
    def __init__(self, encoder, bottleneck, latent_dim, num_classes):
        super(Classifier, self).__init__()

        self.encoder = encoder
        self.bottleneck = bottleneck
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()
        self.classifier = nn.Linear(latent_dim, num_classes)

        self._freeze_encoder()

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters())

    def forward(self, inputs):
        features = self._extract_features(inputs)
        logits = self.classifier(features)

        return logits

    def _extract_features(self, inputs):
        features = self.encoder(inputs)
        if self.bottleneck is not None:
            features, _ = self.bottleneck(features)

        return features

    def training_step(self, batch, batch_idx):
        features, labels = batch
        logits = self(features)
        loss = self.criterion(logits, labels)
        self.log('train/loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        accuracy = self._get_accuracy(batch)
        self.log('val/accuracy', accuracy)

    def test_step(self, batch, batch_idx):
        accuracy = self._get_accuracy(batch)
        self.log('test/accuracy', accuracy)

    def _get_accuracy(self, batch):
        features, labels = batch
        logits = self(features)
        predictions = torch.argmax(logits, dim=1)
        accuracy = torch.sum(predictions == labels).float() / labels.shape[0]

        return accuracy

    def _freeze_encoder(self):
        for m in self.encoder.modules():
            utils.freeze_layer(m)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_encoder()
