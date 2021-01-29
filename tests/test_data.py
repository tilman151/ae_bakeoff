import os
import unittest

from data import MNISTDataModule, FashionMNISTDataModule, KMNISTDataModule


class TestMNISTTemplate:
    DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'data')

    def test_anomaly(self):
        anomaly = 9
        datamodule = self._get_mnist(self.DATA_ROOT, exclude=anomaly)
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
        datamodule = self._get_mnist(self.DATA_ROOT, train_size=train_size)
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
