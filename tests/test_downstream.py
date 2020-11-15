import unittest

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import TensorDataset, DataLoader

from building import build_ae
from data import MNISTDataModule
from downstream.anomaly import AnomalyDetection
from downstream.classification import Classifier
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
        tpr, fpr, thresholds = self.anomaly_detector.get_test_roc(self.data)
        test_dataloader = self.data.test_dataloader()
        scores = self.anomaly_detector.score(test_dataloader)
        self.assertTrue(np.all(thresholds >= np.min(scores)))
        self.assertTrue(np.all(thresholds[1:] <= np.max(scores)))
        self.assertGreaterEqual(thresholds[1], np.max(scores))

    @unittest.skip('Only for visual inspection')
    def test_roc_plotting(self):
        tpr, fpr, thresholds = self.anomaly_detector.get_test_roc(self.data)
        fig = self.anomaly_detector.plot_roc(tpr, fpr)
        fig.show()


class TestClassification(ModelTestsMixin, FrozenLayerCheckMixin, unittest.TestCase):
    def setUp(self):
        encoder = encoders.DenseEncoder((1, 32, 32), 3, 64)
        bottleneck = bottlenecks.VariationalBottleneck(32)
        self.net = Classifier(encoder, bottleneck, 32, 10)
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.output_shape = torch.Size((16, 10))

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
