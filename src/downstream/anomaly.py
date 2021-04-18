import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score

from building import load_ae_from_checkpoint


class AnomalyDetection:
    def __init__(self, autoencoder, num_latent_samples: int = 1):
        self.autoencoder = autoencoder.to('cpu')
        self.autoencoder.eval()
        self.num_latent_samples = num_latent_samples

        self.score_func = torch.nn.BCELoss(reduction='none')

    def get_test_roc(self, datamodule):
        datamodule.prepare_data()
        datamodule.setup('test')
        test_dataloader = datamodule.test_dataloader()
        scores = self.score(test_dataloader)
        anomaly_labels = self.get_test_anomaly_labels(test_dataloader, anomaly_value=datamodule.exclude)
        fpr, tpr, thresholds = roc_curve(anomaly_labels, scores)
        auc = roc_auc_score(anomaly_labels, scores)
        coverages, risks = self._get_coverages_and_risks(scores, anomaly_labels, thresholds)

        return fpr, tpr, thresholds, coverages, risks, auc

    @torch.no_grad()
    def score(self, dataloader):
        scores = []
        for features, _ in dataloader:
            scores.append(self._score_batch(features))
        scores = np.concatenate(scores)

        return scores

    def _score_batch(self, batch):
        batch_size = batch.shape[0]
        reconstruction = self.autoencoder(batch, self.num_latent_samples)
        batch = batch.unsqueeze(1).repeat(1, self.num_latent_samples, 1, 1, 1)
        score = torch.sum(self.score_func(reconstruction, batch).view(batch_size, self.num_latent_samples, -1), dim=-1)
        score = score.mean(dim=1)
        score = score.numpy()

        return score

    def _get_coverages_and_risks(self, scores, labels, thresholds):
        coverages = np.empty_like(thresholds)
        risks = np.empty_like(thresholds)
        for i, thresh in enumerate(thresholds):
            predictions = scores < thresh
            num_covered = np.sum(predictions)
            coverages[i] = num_covered / len(scores)
            risks[i] = np.sum(predictions * labels) / (num_covered + 1e-8)

        return coverages, risks

    @staticmethod
    def get_test_anomaly_labels(dataloader, anomaly_value):
        labels = []
        for _, class_labels in dataloader:
            labels.append(class_labels == anomaly_value)
        labels = torch.cat(labels).numpy()

        return labels

    @classmethod
    def from_autoencoder_checkpoint(cls, model_type, dm, checkpoint_path):
        model = load_ae_from_checkpoint(model_type, dm.dims, anomaly=True, checkpoint_path=checkpoint_path)
        classifier = cls(model)

        return classifier
