import os

import pytorch_lightning as pl

import building
import downstream
import run
from downstream.results import ResultsMixin


class ReproductionRun:
    def __init__(self):
        self.checkpoints = Checkpoints()
        self.classification_results = ClassificationDownstream()
        self.anomaly_detection_results = AnomalyDownstream()
        self.latent_results = LatentDownstream()

    def reproduce(self):
        if self.checkpoints.empty():
            self.train_all()
            self.checkpoints.save()
        for model_type in self.checkpoints.keys():
            self.perform_downstream(model_type)

    def train_all(self):
        for model_type in run.AUTOENCODERS:
            self.checkpoints[model_type] = {}
            self.checkpoints[model_type]['general'] = run.run(model_type)
            self.checkpoints[model_type]['anomaly'] = run.run(model_type, anomaly=True)

    def perform_downstream(self, model_type):
        # self.perform_classification(model_type)
        # self.perform_anomaly_detection(model_type)
        self.perform_latent_tasks(model_type)

    def perform_classification(self, model_type):
        pl.seed_everything(42)
        checkpoint_path = self.checkpoints[model_type]['general']
        self.classification_results.add_accuracy_for(model_type, checkpoint_path)

    def perform_anomaly_detection(self, model_type):
        pl.seed_everything(42)
        checkpoint_path = self.checkpoints[model_type]['anomaly']
        self.anomaly_detection_results.add_roc_for(model_type, checkpoint_path)

    def perform_latent_tasks(self, model_type):
        pl.seed_everything(42)
        checkpoint_path = self.checkpoints[model_type]['general']
        self.latent_results.add_samples_for(model_type, checkpoint_path)
        self.latent_results.add_reconstructions_for(model_type, checkpoint_path)
        self.latent_results.add_interpolation_for(model_type, checkpoint_path)
        self.latent_results.add_reduction_for(model_type, checkpoint_path)


class Checkpoints(ResultsMixin):
    def _get_results_path(self):
        log_path = self._get_log_path()
        checkpoint_path = os.path.join(log_path, 'checkpoints.json')

        return checkpoint_path


class ClassificationDownstream(ResultsMixin):
    def add_accuracy_for(self, model_type, checkpoint_path):
        accuracy = self._get_test_accuracy(model_type, checkpoint_path)
        self[model_type] = accuracy
        self.save()

    def _get_test_accuracy(self, model_type, checkpoint_path):
        data = building.build_datamodule('classification')
        classifier = downstream.Classifier.from_autoencoder_checkpoint(model_type, data, checkpoint_path)
        trainer = self._get_classification_trainer()

        trainer.fit(classifier, datamodule=data)
        test_results, *_ = trainer.test(datamodule=data)
        accuracy = test_results['test/accuracy']

        return accuracy

    def _get_classification_trainer(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint('val/accuracy', mode='max')
        early_stop_callback = pl.callbacks.EarlyStopping('val/accuracy', mode='max')
        trainer = pl.Trainer(logger=False,
                             max_epochs=20,
                             checkpoint_callback=checkpoint_callback,
                             early_stop_callback=early_stop_callback)

        return trainer

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'classification_results.json')

        return results_path


class AnomalyDownstream(ResultsMixin):
    def add_roc_for(self, model_type, checkpoint_path):
        fpr, tpr, thresholds, auc = self._get_test_roc(model_type, checkpoint_path)
        self[model_type] = {'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'thresholds': thresholds.tolist(),
                            'auc': auc}
        self.save()

    def _get_test_roc(self, model_type, checkpoint_path):
        data = building.build_datamodule(anomaly=True)
        anomaly_detector = downstream.AnomalyDetection.from_autoencoder_checkpoint(model_type, data, checkpoint_path)
        fpr, tpr, thresholds, auc = anomaly_detector.get_test_roc(data)

        return fpr, tpr, thresholds, auc

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'anomaly_results.json')

        return results_path


class LatentDownstream(ResultsMixin):
    def add_samples_for(self, model_type, checkpoint_path):
        data = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data, checkpoint_path)
        samples = latent_sampler.sample(16)
        if samples is not None:
            self._save_samples(model_type, samples)

    def _save_samples(self, model_type, samples):
        self.save_image_result(model_type, 'samples', samples)

    def add_reconstructions_for(self, model_type, checkpoint_path):
        data = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data, checkpoint_path)
        batch = self._get_data_batch(data, n=16)
        reconstructions = latent_sampler.reconstruct(batch)
        self._save_reconstructions(model_type, reconstructions)

    def _save_reconstructions(self, model_type, reconstructions):
        self.save_image_result(model_type, 'reconstructions', reconstructions)

    def add_interpolation_for(self, model_type, checkpoint_path):
        data = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data, checkpoint_path)
        start, end = self._get_data_batch(data, n=2)
        interpolation = latent_sampler.interpolate(start, end, steps=128)
        self._save_interpolation(model_type, interpolation)

    def _save_interpolation(self, model_type, interpolation):
        self.save_video_result(model_type, 'interpolation', interpolation)

    def add_reduction_for(self, model_type, checkpoint_path):
        data = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data, checkpoint_path)
        reduction, labels = latent_sampler.reduce(data.test_dataloader())
        self._save_reduction(model_type, reduction, labels)

    def _save_reduction(self, model_type, reduction, labels):
        self.save_array_result(model_type, 'reduction', reduction, labels)

    def _get_datamodule(self):
        data = building.build_datamodule()
        data.prepare_data()
        data.setup('test')
        return data

    def _get_data_batch(self, data, n):
        test_loader = data.test_dataloader()
        batch, _ = next(iter(test_loader))
        batch = batch[:n]

        return batch

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'latent_results.json')

        return results_path


if __name__ == '__main__':
    ReproductionRun().reproduce()
