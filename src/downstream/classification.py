import pytorch_lightning as pl
import torch
from torch import nn as nn

import utils


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
