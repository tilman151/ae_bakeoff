"""Adopted from https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html#lightningdatamodule-api"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset

from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', apply_noise=False, train_size=None, exclude=None):
        super().__init__()
        self.data_dir = data_dir
        self.transform = [transforms.Pad(2),
                          transforms.ToTensor()]

        self.dims = (1, 32, 32)
        self.num_classes = 10 if exclude is None else 9
        self.apply_noise = apply_noise
        self.train_size = train_size
        self.exclude = exclude

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        transform = self._get_transforms(stage)
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=transform)
            self.mnist_train, self.mnist_val = self._split_train_val(mnist_full)

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=transform)

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

    def _get_transforms(self, stage):
        if self.apply_noise and stage == 'fit':
            transform = self.transform + [AddNoise(noise_ratio=0.1)]
        else:
            transform = self.transform

        return transforms.Compose(transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=0)


class AddNoise:
    def __init__(self, noise_ratio):
        self.noise_ratio = noise_ratio

    def __call__(self, img):
        img = img + torch.randn_like(img) * self.noise_ratio
        img = torch.clamp(img, min=0., max=1.)

        return img
