import unittest
from unittest import mock

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

from building import build_ae
from data import MNISTDataModule
from downstream import AnomalyDetection, Classifier, Latent
from downstream import formatting
from models import encoders, bottlenecks
from tests.templates import ModelTestsMixin, FrozenLayerCheckMixin


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
        tpr, fpr, thresholds, auc = self.anomaly_detector.get_test_roc(self.data)
        test_dataloader = self.data.test_dataloader()
        scores = self.anomaly_detector.score(test_dataloader)
        self.assertTrue(np.all(thresholds >= np.min(scores)))
        self.assertTrue(np.all(thresholds[1:] <= np.max(scores)))
        self.assertGreaterEqual(thresholds[1], np.max(scores))
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
        self._check_frozen(self.net.encoder)

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
        batch, _ = next(iter(self.data.test_dataloader()))
        comparison = self.latent_viz.reconstruct(batch)
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
