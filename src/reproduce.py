import os

import matplotlib.pyplot as plt
import numpy as np
import pytablewriter
import pytorch_lightning as pl

import building
import data
import downstream
import run
import utils
from downstream.results import AbstractResults


class ReproductionRun:
    def __init__(self, dataset, retrain, recalc_downstream, batch_size, gpu):
        load_checkpoints = not retrain
        load_downstream_results = (not recalc_downstream) and load_checkpoints

        self.dataset = dataset
        self.batch_size = batch_size
        self.gpu = gpu

        self.checkpoints = Checkpoints(dataset, load_checkpoints)
        self.training_time_results = TrainingTimeResults(dataset, load_downstream_results)
        self.classification_results = ClassificationDownstream(dataset, load_downstream_results)
        self.anomaly_detection_results = AnomalyDownstream(dataset, load_downstream_results)
        self.latent_results = LatentDownstream(dataset, load_downstream_results)
        self.latent_anomaly_results = LatentDownstream(dataset, load_downstream_results, tag='anomaly')
        self.reconstruction_results = ReconstructionResults(dataset, load_downstream_results)

    def reproduce(self):
        if self.checkpoints.empty():
            self.train_all()
            self.checkpoints.save()
        for model_type in self.checkpoints.keys():
            print(f'Perform downstream tasks for {model_type}...')
            self.perform_downstream(model_type)
        self.render_results()

    def train_all(self):
        for model_type in run.AUTOENCODERS:
            self.checkpoints[model_type] = {}
            self.checkpoints[model_type]['general'] = run.run(model_type,
                                                              self.dataset,
                                                              self.batch_size,
                                                              self.gpu)
            self.checkpoints[model_type]['anomaly'] = run.run(model_type,
                                                              self.dataset,
                                                              self.batch_size,
                                                              self.gpu,
                                                              anomaly=True)

    def perform_downstream(self, model_type):
        self.perform_training_time(model_type)
        self.perform_classification(model_type)
        self.perform_anomaly_detection(model_type)
        self.perform_latent_tasks(model_type)
        self.perform_reconstruction(model_type)

    def perform_training_time(self, model_type):
        print("Extract Training Time...")
        if model_type not in self.training_time_results:
            checkpoint_path = self.checkpoints[model_type]['general']
            self.training_time_results.add_training_time_for(model_type, checkpoint_path)

    def perform_classification(self, model_type):
        print('Classification...')
        if model_type not in self.classification_results:
            pl.seed_everything(42)
            checkpoint_path = self.checkpoints[model_type]['general']
            self.classification_results.add_accuracy_for(model_type, checkpoint_path)

    def perform_anomaly_detection(self, model_type):
        print('Anomaly Detection...')
        if model_type not in self.anomaly_detection_results:
            pl.seed_everything(42)
            checkpoint_path = self.checkpoints[model_type]['anomaly']
            self.anomaly_detection_results.add_roc_for(model_type, checkpoint_path)

    def perform_latent_tasks(self, model_type):
        print('Latent Space Visualization...')
        if model_type not in self.latent_results:
            pl.seed_everything(42)
            checkpoint_path = self.checkpoints[model_type]['general']
            self._perform_all_latent(model_type, checkpoint_path)
            checkpoint_path = self.checkpoints[model_type]['anomaly']
            self.latent_anomaly_results.add_reduction_for(model_type, checkpoint_path)

    def _perform_all_latent(self, model_type, checkpoint_path):
        self.latent_results.add_samples_for(model_type, checkpoint_path)
        self.latent_results.add_interpolation_for(model_type, checkpoint_path)
        self.latent_results.add_reduction_for(model_type, checkpoint_path)

    def perform_reconstruction(self, model_type):
        print('Reconstruction...')
        if model_type not in self.reconstruction_results:
            pl.seed_everything(42)
            checkpoint_path = self.checkpoints[model_type]['general']
            self.reconstruction_results.add_reconstructions_for(model_type, checkpoint_path)

    def render_results(self):
        self.training_time_results.render()
        self.classification_results.render()
        self.anomaly_detection_results.render()
        self.latent_results.render()
        self.latent_anomaly_results.render()
        self.reconstruction_results.render()


class Checkpoints(AbstractResults):
    def _get_output_path(self):
        pass

    def render(self):
        pass

    def _get_results_path(self):
        log_path = self._get_log_path()
        checkpoint_path = os.path.join(log_path, 'checkpoints.json')

        return checkpoint_path


class ClassificationDownstream(AbstractResults):
    def add_accuracy_for(self, model_type, checkpoint_path):
        accuracy = self._get_test_accuracy(model_type, checkpoint_path)
        self[model_type] = accuracy
        self.save()

    def _get_test_accuracy(self, model_type, checkpoint_path):
        data_module = building.build_datamodule(self.dataset, 'classification')
        classifier = downstream.Classifier.from_autoencoder_checkpoint(model_type, data_module, checkpoint_path)
        trainer = self._get_classification_trainer()

        trainer.fit(classifier, datamodule=data_module)
        test_results, *_ = trainer.test(datamodule=data_module)
        accuracy = test_results['test/accuracy']

        return accuracy

    def _get_classification_trainer(self):
        tmp_checkpoint_dir = utils.tempdir()
        checkpoint_callback = pl.callbacks.ModelCheckpoint(tmp_checkpoint_dir, 'val/accuracy', mode='max')
        early_stop_callback = pl.callbacks.EarlyStopping('val/accuracy', mode='max')
        trainer = pl.Trainer(logger=False,
                             max_epochs=20,
                             checkpoint_callback=checkpoint_callback,
                             early_stop_callback=early_stop_callback,
                             progress_bar_refresh_rate=0)

        return trainer

    def render(self):
        print('Render classification table...')
        markdown_table = pytablewriter.MarkdownTableWriter(table_name='Classification Results',
                                                           headers=list(self.keys()),
                                                           value_matrix=[list(self.values())])
        markdown_file = self._get_output_path()
        markdown_table.dump(markdown_file)

    def _get_output_path(self):
        log_path = self._get_log_path()
        output_path = os.path.join(log_path, 'classification.md')

        return output_path

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'classification_results.json')

        return results_path


class AnomalyDownstream(AbstractResults):
    def add_roc_for(self, model_type, checkpoint_path):
        fpr, tpr, thresholds, auc = self._get_test_roc(model_type, checkpoint_path)
        self[model_type] = {'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'thresholds': thresholds.tolist(),
                            'auc': auc}
        self.save()

    def _get_test_roc(self, model_type, checkpoint_path):
        data_module = building.build_datamodule(self.dataset, anomaly=True)
        anomaly_detector = downstream.AnomalyDetection.from_autoencoder_checkpoint(model_type, data_module, checkpoint_path)
        fpr, tpr, thresholds, auc = anomaly_detector.get_test_roc(data_module)

        return fpr, tpr, thresholds, auc

    def render(self):
        print('Render anomaly detection results...')
        num_subplots = len(self.keys())
        fig, axes = utils.get_axes_grid(num_subplots, ncols=3, ax_size=4)
        self._plot_rocs(axes)
        fig.tight_layout()
        plt.savefig(self._get_output_path())
        plt.close()

    def _plot_rocs(self, axes):
        for ax, model_type in zip(axes, self.keys()):
            downstream.plot_roc(ax,
                                self[model_type]['fpr'],
                                self[model_type]['tpr'],
                                self[model_type]['auc'],
                                title=model_type)

    def _get_output_path(self):
        log_path = self._get_log_path()
        output_path = os.path.join(log_path, 'anomaly.png')

        return output_path

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'anomaly_results.json')

        return results_path


class LatentDownstream(AbstractResults):
    def add_samples_for(self, model_type, checkpoint_path):
        datamodule = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, datamodule, checkpoint_path)
        samples = latent_sampler.sample(16)
        if samples is not None:
            self._save_samples(model_type, samples)

    def _save_samples(self, model_type, samples):
        self.save_image_result(model_type, f'samples{self._get_tag_suffix()}', samples)

    def add_interpolation_for(self, model_type, checkpoint_path):
        data_module = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data_module, checkpoint_path)
        start, end = self._get_start_end_frames(data_module)
        interpolation = latent_sampler.interpolate(start, end, steps=128)
        self._save_interpolation(model_type, interpolation)

    def _get_start_end_frames(self, data_module):
        test_loader = data_module.test_dataloader()
        batch, _ = next(iter(test_loader))
        start, end = batch[:2]

        return start, end

    def _save_interpolation(self, model_type, interpolation):
        self.save_video_result(model_type, f'interpolation{self._get_tag_suffix()}', interpolation)

    def add_reduction_for(self, model_type, checkpoint_path):
        data_module = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data_module, checkpoint_path)
        reduction, labels = latent_sampler.reduce(data_module.test_dataloader())
        self._save_reduction(model_type, reduction, labels)

    def _save_reduction(self, model_type, reduction, labels):
        self.save_array_result(model_type, f'reduction{self._get_tag_suffix()}', reduction, labels)

    def _get_datamodule(self):
        data_module = building.build_datamodule(self.dataset)
        data_module.prepare_data()
        data_module.setup('test')

        return data_module

    def render(self):
        print(f'Render reduction{self._get_tag_suffix()} results...')
        num_subplots = len(self.keys())
        fig, axes = utils.get_axes_grid(num_subplots, ncols=3, ax_size=6)
        self._plot_reductions(axes)
        self._make_legend(fig, axes)
        fig.tight_layout()
        plt.savefig(self._get_output_path())
        plt.close()

    def _plot_reductions(self, axes):
        for ax, model_type in zip(axes, self.keys()):
            features, labels = self._load_reduction(model_type)
            downstream.plot_reduction(ax, features, labels, model_type)

    def _load_reduction(self, model_type):
        file_path = self[model_type][f'reduction{self._get_tag_suffix()}']
        data_module = np.load(file_path)
        features = data_module['arr_0']
        labels = data_module['arr_1']

        return features, labels

    def _make_legend(self, fig, axes):
        handels, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handels, labels)
        for handle in legend.legendHandles:
            handle._sizes = [20]
            handle._alpha = 1

    def _get_output_path(self):
        log_path = self._get_log_path()
        output_path = os.path.join(log_path, f'reduction{self._get_tag_suffix()}.png')

        return output_path

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, f'latent_results{self._get_tag_suffix()}.json')

        return results_path

    def _get_tag_suffix(self):
        return '' if self.tag is None else f'_{self.tag}'


class ReconstructionResults(AbstractResults):
    def add_reconstructions_for(self, model_type, checkpoint_path):
        data_module = self._get_datamodule()
        latent_sampler = downstream.Latent.from_autoencoder_checkpoint(model_type, data_module, checkpoint_path)
        loss, reconstructions = latent_sampler.reconstruct(data_module, num_comparison=16)
        self._save_reconstructions(model_type, loss, reconstructions)

    def _get_datamodule(self):
        data_module = building.build_datamodule(self.dataset)
        data_module.prepare_data()
        data_module.setup('test')

        return data_module

    def _save_reconstructions(self, model_type, loss, reconstructions):
        self.safe_add(model_type, 'loss', loss)
        self.save_image_result(model_type, 'reconstructions', reconstructions)

    def render(self):
        print('Render reconstruction table...')
        values = [v['loss'] for v in self.values()]
        markdown_table = pytablewriter.MarkdownTableWriter(table_name='Reconstruction Results',
                                                           headers=list(self.keys()),
                                                           value_matrix=[values])
        markdown_file = self._get_output_path()
        markdown_table.dump(markdown_file)

    def _get_output_path(self):
        log_path = self._get_log_path()
        output_path = os.path.join(log_path, 'reconstruction.md')

        return output_path

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'reconstruction_results.json')

        return results_path


class TrainingTimeResults(AbstractResults):
    def add_training_time_for(self, model_type, checkpoint_path):
        event_file_path = self._get_event_file_path(checkpoint_path)
        time_extractor = downstream.TrainingTime(event_file_path)
        training_time = time_extractor.get_training_time()
        self.results[model_type] = str(training_time)
        self.save()

    def _get_event_file_path(self, checkpoint_path):
        return os.path.dirname(os.path.dirname(checkpoint_path))

    def _get_results_path(self):
        log_path = self._get_log_path()
        results_path = os.path.join(log_path, 'training_time_results.json')

        return results_path

    def render(self):
        print('Render training time table...')
        markdown_table = pytablewriter.MarkdownTableWriter(table_name='Training Times',
                                                           headers=list(self.keys()),
                                                           value_matrix=[list(self.values())])
        markdown_file = self._get_output_path()
        markdown_table.dump(markdown_file)

    def _get_output_path(self):
        log_path = self._get_log_path()
        output_path = os.path.join(log_path, 'training_time.md')

        return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reproduce all results')
    parser.add_argument('--dataset', default='mnist', choices=data.AVAILABLE_DATASETS.keys())
    parser.add_argument('--retrain', action='store_true', help='Retrain even if checkpoints are available')
    parser.add_argument('--recalc_downstream', action='store_true', help='Recalculate downstream tasks')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    opt = parser.parse_args()

    ReproductionRun(opt.dataset, opt.retrain, opt.recalc_downstream, opt.batch_size, opt.gpu).reproduce()
