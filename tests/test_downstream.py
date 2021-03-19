import datetime
import unittest
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
from building import build_ae
from data import MNISTDataModule
from downstream import AnomalyDetection, Classifier, Latent
from downstream import formatting
from downstream.time import TrainingTime
from models import encoders, bottlenecks
from templates import ModelTestsMixin, FrozenLayerCheckMixin


class TestAnomalyDetection(unittest.TestCase):
    def setUp(self):
        self.data = MNISTDataModule(data_dir='../data', exclude=9)
        self.data.prepare_data()
        self.data.setup()
        self.net = build_ae('vanilla', self.data.dims)
        self.anomaly_detector = AnomalyDetection(self.net)

    def test_anomaly_labels(self):
        class_labels = torch.randint(0, 10, size=(100,))
        dummy_features = torch.randn(100, 50)
        dummy_dataset = TensorDataset(dummy_features, class_labels)

        expected_anomaly_labels = (class_labels == 9).numpy().astype(np.int)
        actual_anomaly_labels = self.anomaly_detector.get_test_anomaly_labels(DataLoader(dummy_dataset, batch_size=32),
                                                                              anomaly_value=9)
        self.assertListEqual(expected_anomaly_labels.tolist(), actual_anomaly_labels.tolist())

    def test_score(self):
        test_dataloader = self.data.test_dataloader()
        scores = self.anomaly_detector.score(test_dataloader)
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(10000, scores.shape[0])  # number of scores is length of test data

    def test_roc(self):
        tpr, fpr, thresholds, coverages, auc, accuracies = self.anomaly_detector.get_test_roc(self.data)
        test_dataloader = self.data.test_dataloader()
        scores = self.anomaly_detector.score(test_dataloader)
        self.assertTrue(np.all(thresholds >= np.min(scores)))
        self.assertTrue(np.all(thresholds[1:] <= np.max(scores)))
        self.assertGreaterEqual(thresholds[1], np.max(scores))
        self.assertEqual(1., coverages[0])
        self.assertEqual(0., coverages[-1])
        self.assertLessEqual(0, auc)
        self.assertGreaterEqual(1, auc)


class TestClassification(ModelTestsMixin, FrozenLayerCheckMixin, unittest.TestCase):
    def setUp(self):
        encoder = encoders.DenseEncoder((1, 32, 32), 3, 64)
        bottleneck = bottlenecks.VariationalBottleneck(32)
        self.net = Classifier(encoder, bottleneck, 10)
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.output_shape = torch.Size((16, 10))

    def test_latent_dim(self):
        self.assertEqual(self.net.bottleneck.latent_dim, self.net.latent_dim)

    def test_accuracy(self):
        accuracy = self.net._get_accuracy((self.test_inputs, torch.zeros(self.test_inputs.shape[0])))
        self.assertLessEqual(0, accuracy)
        self.assertGreaterEqual(1, accuracy)

    def test_layers_frozen(self):
        with self.subTest('should_be_frozen'):
            self._check_frozen(self.net.encoder)
        with self.subTest('should_be_unfrozen'):
            self.net.freeze_encoder = False
            self.net.encoder = encoders.DenseEncoder((1, 32, 32), 3, 64)
            self.net.train()
            self._check_frozen(self.net.encoder, should_be_frozen=False)

    def test_accuracy_returned_on_test(self):
        datamodule = MNISTDataModule(data_dir='../data')
        trainer = pl.Trainer(logger=False)
        test_results, *_ = trainer.test(self.net, datamodule=datamodule)
        self.assertIsNotNone(test_results)
        self.assertLessEqual(0, test_results['test/accuracy'])
        self.assertGreaterEqual(1, test_results['test/accuracy'])


class TestLatent(unittest.TestCase):
    def setUp(self):
        self.data = MNISTDataModule(data_dir='../data')
        self.data.prepare_data()
        self.data.setup()
        self.net = build_ae('vae', self.data.dims)
        self.latent_viz = Latent(self.net)

    def test_sample(self):
        samples = self.latent_viz.sample(16)
        expected_shape = torch.Size((3, 2 * 32 + 3 * 2, 8 * 32 + 9 * 2))  # 2 rows x 8 cols + 2 pad each side
        self.assertEqual(expected_shape, samples.shape)

    def test_reconstruct(self):
        _, comparison = self.latent_viz.reconstruct(self.data, num_comparison=32)
        expected_shape = torch.Size((3, 32 * 32 + 33 * 2, 2 * 32 + 3 * 2))  # 32 rows x 2 cols + 2 pad each side
        self.assertEqual(expected_shape, comparison.shape)

    def test_interpolate(self):
        data_iter = iter(self.data.test_dataloader())
        start_batch, _ = next(data_iter)
        end_batch, _ = next(data_iter)
        steps = 1
        interpolations = self.latent_viz.interpolate(start_batch, end_batch, steps=steps)

        expected_shape = (steps + 2, 3, 4 * 32 + 5 * 2, 8 * 32 + 9 * 2)  # 3 steps x 4 rows x 8 cols + 2 pad each side
        self.assertEqual(expected_shape, interpolations.shape)

    def test_reduce(self):
        reduced_latents, labels = self.latent_viz.reduce(self.data.val_dataloader())
        self.assertEqual(len(self.data.mnist_val), reduced_latents.shape[0])
        self.assertEqual(len(self.data.mnist_val), labels.shape[0])


class TestFormatting(unittest.TestCase):
    @mock.patch('downstream.formatting.Image.Image.save')
    def test_save_imagegrid(self, mock_write_jpeg):
        imagegrid = np.random.random((3, 500, 500))
        file_name = 'test.jpeg'
        formatting.save_imagegrid(imagegrid, file_name)
        mock_write_jpeg.assert_called_with(file_name)

    @mock.patch('downstream.formatting.Image.Image.save')
    def test__save_gif(self, mock_save):
        vid = np.random.random((2, 32, 32, 3)) * 255
        vid = vid.astype(np.uint8)
        file_name = 'test.gif'
        formatting._save_gif(vid, file_name, 5, False)

    @mock.patch('downstream.formatting._save_gif')
    def test_save_video(self, mock_write_video):
        vid = np.random.random((100, 3, 32, 32))
        file_name = 'test.mp4'
        formatting.save_video(vid, file_name)

        written_tensor = mock_write_video.call_args[0][0]
        self.assertGreaterEqual(255, written_tensor.max())
        self.assertLessEqual(0, written_tensor.min())
        self.assertEqual((100, 32, 32, 3), written_tensor.shape)

    @mock.patch('downstream.formatting.save_video')
    def test_save_oscillating_video(self, mock_write_video):
        vid = np.random.random((100, 3, 32, 32))
        file_name = 'test.mp4'
        formatting.save_oscillating_video(vid, file_name)

        written_tensor = mock_write_video.call_args[0][0]
        self.assertGreater(written_tensor.shape[0], vid.shape[0] * 2)
        self.assertEqual(0, np.sum(written_tensor[0] - vid[0]))
        self.assertEqual(0, np.sum(written_tensor[-1] - vid[0]))

    @unittest.skip('Only for visual inspection')
    def test_roc_plotting(self):
        fig = plt.figure(figsize=(5, 5))
        tpr = np.linspace(0, 1, num=50)
        fpr = np.linspace(0, 1, num=50)
        auc = 0.5
        formatting.plot_roc(plt.gca(), tpr, fpr, auc)
        fig.show()

    @unittest.skip('Only for visual inspection')
    def test_coverage_plotting(self):
        fig = plt.figure(figsize=(5, 5))
        coverage = np.linspace(0, 1, num=50)
        accuracy = np.linspace(0.2, 0.5, num=50)
        formatting.plot_perfect_risk_coverage(plt.gca())
        formatting.plot_risk_coverage(plt.gca(), coverage, accuracy, "test")
        plt.legend(loc='lower right')
        fig.show()


class TestTrainingTime(unittest.TestCase):
    def setUp(self):
        self.temp_dir = utils.tempdir()
        self.num_events = 100
        self.start_time = datetime.datetime.now()
        self.end_time = self.start_time + datetime.timedelta(seconds=self.num_events)
        self.summary_writer = SummaryWriter(self.temp_dir, filename_suffix="tmp")
        self.event_file_path = self.summary_writer.file_writer.event_writer._file_name

    def test_creation(self):
        with self.assertRaises(ValueError):
            TrainingTime("bogus/path")
        with self.assertRaises(RuntimeError):
            TrainingTime(self.event_file_path)
        self._add_events()
        TrainingTime(self.event_file_path)

    def test_training_time_calculation(self):
        self._add_events()
        training_time = TrainingTime(self.event_file_path)
        expected_training_time = self.end_time - self.start_time
        actual_training_time = training_time.get_training_time()
        self.assertEqual(expected_training_time, actual_training_time)

    def _add_events(self):
        for i in range(self.num_events):
            train_time = self.start_time + datetime.timedelta(seconds=i)
            self.summary_writer.add_scalar(TrainingTime._TRAIN_TAG, i, global_step=i, walltime=train_time.timestamp())
            val_time = self.start_time + datetime.timedelta(seconds=i+1)
            self.summary_writer.add_scalar(TrainingTime._VAL_TAG, i, global_step=i, walltime=val_time.timestamp())
        self.summary_writer.flush()
