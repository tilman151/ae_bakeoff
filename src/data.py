"""Adopted from https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html#lightningdatamodule-api"""

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './'):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.Pad(2),
            transforms.ToTensor()
        ])

        self.dims = (1, 32, 32)

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000],
                                                            generator=torch.Generator().manual_seed(42))

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32, num_workers=2)
