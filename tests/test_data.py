import os
import unittest
from unittest import mock

from PIL import ImageChops
from torchvision.datasets import MNIST, EMNIST

from data import MNISTDataModule, FashionMNISTDataModule, KMNISTDataModule, MNISTWithEMNISTTest, \
    MNISTWithEMNISTTestDataModule

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')


class TestMNISTTemplate:
    def test_anomaly(self):
        anomaly = 9
        datamodule = self._get_mnist(DATA_ROOT, exclude=anomaly)
        datamodule.prepare_data()
        datamodule.setup()
        with self.subTest(split='train'):
            self._check_excluded(anomaly, datamodule.train_dataloader())
        with self.subTest(split='val'):
            self._check_included(anomaly, datamodule.val_dataloader())
        with self.subTest(split='test'):
            self._check_included(anomaly, datamodule.test_dataloader())

    def _check_excluded(self, excluded, loader):
        for _, labels in loader:
            self.assertNotIn(excluded, labels)

    def _check_included(self, included, loader):
        for _, labels in loader:
            if included in labels:
                return
        raise AssertionError(f'{included} not found in any batch.')

    def test_train_size(self):
        train_size = 500
        datamodule = self._get_mnist(DATA_ROOT, train_size=train_size)
        datamodule.prepare_data()
        datamodule.setup()
        train_data = datamodule.train_dataloader().dataset
        self.assertEqual(train_size, len(train_data))

    def _get_mnist(self, data_root, batch_size=32, exclude=None, train_size=None):
        raise NotImplementedError


class TestMNIST(unittest.TestCase, TestMNISTTemplate):
    def _get_mnist(self, data_root, batch_size=32, exclude=None, train_size=None):
        return MNISTDataModule(data_root, batch_size, train_size, exclude)


class TestFashionMNIST(unittest.TestCase, TestMNISTTemplate):
    def _get_mnist(self, data_root, batch_size=32, exclude=None, train_size=None):
        return FashionMNISTDataModule(data_root, batch_size, train_size, exclude)


class TestKMNIST(unittest.TestCase, TestMNISTTemplate):
    def _get_mnist(self, data_root, batch_size=32, exclude=None, train_size=None):
        return KMNISTDataModule(data_root, batch_size, train_size, exclude)


class TestEMNIST(unittest.TestCase, TestMNISTTemplate):
    def _get_mnist(self, data_root, batch_size=32, exclude=None, train_size=None):
        return MNISTWithEMNISTTestDataModule(data_root, batch_size, train_size, exclude)


class TestMNISTWithEMNISTTest(unittest.TestCase):
    def test_get_item(self):
        with self.subTest("train"):
            dataset = MNISTWithEMNISTTest(DATA_ROOT, train=True, download=True)
            train_label_range = range(10)
            for _, label in dataset:
                self.assertIn(label, train_label_range)

        with self.subTest("test"):
            dataset = MNISTWithEMNISTTest(DATA_ROOT, train=False, download=True)
            test_label_range = range(11)
            for _, label in dataset:
                self.assertIn(label, test_label_range)

    def test_test_data_ratio(self):
        dataset = MNISTWithEMNISTTest(DATA_ROOT, train=False, download=True)
        labels = [label for _, label in dataset]
        num_emnist = len([label for label in labels if label == 10])
        num_mnist = len([label for label in labels if label < 10])
        self.assertEqual(num_mnist, num_emnist)

    @mock.patch("torch.randperm", return_value=range(6000))
    def test_labeled_correctly(self, _):
        dataset = MNISTWithEMNISTTest(DATA_ROOT, train=False, download=True)

        mnist = MNIST(DATA_ROOT, train=False, download=True)
        for i in range(dataset.half_test_len):
            mnist_feature, mnist_label = mnist[i]
            feature, label = dataset[i]
            self.assertEqual(mnist_label, label)
            diff = ImageChops.difference(mnist_feature, feature)
            self.assertFalse(diff.getbbox())

        emnist = EMNIST(DATA_ROOT, split="letters", train=False, download=True)
        for i in range(dataset.half_test_len):
            emnist_feature, _ = emnist[i]
            feature, label = dataset[dataset.half_test_len + i]
            self.assertEqual(10, label)
            diff = ImageChops.difference(emnist_feature, feature)
            self.assertFalse(diff.getbbox())
