import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from lightning import Autoencoder


class AnomalyDetection:
    def __init__(self, autoencoder):
        self.autoencoder = autoencoder.to('cpu')
        self.autoencoder.eval()

    def get_test_roc(self, datamodule):
        test_dataloader = datamodule.test_dataloader()
        scores = self.score(test_dataloader)
        anomaly_labels = self.get_test_anomaly_labels(test_dataloader, anomaly_value=datamodule.exclude)
        fpr, tpr, thresholds = roc_curve(anomaly_labels, scores)
        auc = roc_auc_score(anomaly_labels, scores)

        return fpr, tpr, thresholds, auc

    @torch.no_grad()
    def score(self, dataloader):
        scores = []
        for features, _ in dataloader:
            scores.append(self._score_batch(features))
        scores = np.concatenate(scores)

        return scores

    def _score_batch(self, batch):
        batch_size = batch.shape[0]
        reconstruction = self.autoencoder(batch)
        score = torch.sum((reconstruction - batch).view(batch_size, -1), dim=1)
        score = score.numpy()

        return score

    @staticmethod
    def get_test_anomaly_labels(dataloader, anomaly_value):
        labels = []
        for _, class_labels in dataloader:
            labels.append(class_labels == anomaly_value)
        labels = torch.cat(labels).numpy()

        return labels

    @classmethod
    def from_autoencoder_checkpoint(cls, checkpoint_path):
        model = Autoencoder.load_from_checkpoint(checkpoint_path, map_location='cpu')
        classifier = cls(model)

        return classifier
