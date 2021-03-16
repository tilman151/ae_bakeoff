"""Adopted from https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html#lightningdatamodule-api"""
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, EMNIST


class MNISTDerivativeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size=32, train_size=None, exclude=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Pad(2),
                                             transforms.ToTensor()])

        self.dims = (1, 32, 32)
        self.num_classes = 10 if exclude is None else 9
        self.batch_size = batch_size
        self.train_size = train_size
        self.exclude = exclude
        self.num_workers = os.cpu_count()

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def _get_mnist(self, train, transform=None, download=False):
        raise NotImplementedError

    def prepare_data(self):
        self._get_mnist(train=True, download=True)
        self._get_mnist(train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = self._get_mnist(train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = self._split_train_val(mnist_full)

        if stage == 'test' or stage is None:
            self.mnist_test = self._get_mnist(train=False, transform=self.transform)

    def _split_train_val(self, mnist_full):
        filter_mask = torch.zeros(len(mnist_full), dtype=torch.int)
        split_idx = torch.randperm(len(mnist_full), generator=torch.Generator().manual_seed(42))
        bootstrap_size = self.train_size if self.train_size is not None else 55000
        filter_mask.scatter_(0, split_idx[:bootstrap_size], 1)
        if self.exclude is not None:
            filter_mask[mnist_full.targets == self.exclude] = 0

        mnist_train = Subset(mnist_full, filter_mask.nonzero(as_tuple=False).squeeze())
        mnist_val = Subset(mnist_full, split_idx[55000:])

        return mnist_train, mnist_val

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers)


class MNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        return MNIST(self.data_dir, train=train, transform=transform, download=download)


class FashionMNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        return FashionMNIST(self.data_dir, train=train, transform=transform, download=download)


class KMNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        return KMNIST(self.data_dir, train=train, transform=transform, download=download)


class MNISTWithEMNISTTest(Dataset):
    def __init__(self, data_dir, train=True, transform=None, target_transform=None, download=False):
        mnist = MNIST(data_dir, train, transform, target_transform, download)
        emnist = EMNIST(data_dir,
                        "letters",
                        train=train,
                        transform=transform,
                        target_transform=target_transform,
                        download=download)

        self.train = train
        self.half_test_len = int(len(mnist) * 0.5)
        if train:
            self.dataset = mnist
        else:
            mnist_part = self._get_random_split(mnist, self.half_test_len)
            emnist_part = self._get_random_split(emnist, self.half_test_len)
            self.dataset = ConcatDataset([mnist_part, emnist_part])

    def _get_random_split(self, dataset, size):
        mnist_idx = torch.randperm(len(dataset))[:size]
        split = Subset(dataset, mnist_idx)

        return split

    def __getitem__(self, index):
        features, label = self.dataset[index]
        if not self.train and index >= self.half_test_len:
            label = 10  # Treat EMNIST samples in test as 11th class

        return features, label

    def __len__(self):
        return len(self.dataset)


class MNISTWithEMNISTTestDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        return MNISTWithEMNISTTest(self.data_dir, train=train, transform=transform, download=download)


AVAILABLE_DATASETS = {'mnist': MNISTDataModule,
                      'fmnist': FashionMNISTDataModule,
                      'kmnist': KMNISTDataModule,
                      'emnist': MNISTWithEMNISTTestDataModule}
