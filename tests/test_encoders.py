import unittest

import torch

from models import encoders
from templates import ModelTestsMixin, FrozenLayerCheckMixin


class TestDenseEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.output_shape = torch.Size((16, 32))
        self.net = encoders.DenseEncoder(self.test_inputs.shape[1:], 3, 32)


class TestShallowEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.output_shape = torch.Size((16, 32))
        self.net = encoders.ShallowEncoder(self.test_inputs.shape[1:], 32)


class TestStackedEncoder(unittest.TestCase, FrozenLayerCheckMixin):
    def setUp(self):
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.net = encoders.StackedEncoder(self.test_inputs.shape[1:], 3, 32)

    def test_stacking(self):
        for i, num_features in enumerate([341, 113, 32]):
            with self.subTest(stack_level=i):
                outputs = self.net(self.test_inputs)
                self.assertEqual(torch.Size((16, num_features)), outputs.shape)
            self.net.stack_layer()

    def test_freezing(self):
        for i in range(3):
            with self.subTest(stack_level=i):
                self._check_frozen(self.net.layers[:i])
                self._check_frozen(self.net.layers[i:], should_be_frozen=False)
            self.net.stack_layer()

    def test_set_training(self):
        self.net.stack_layer()
        self.net.eval()
        self._check_frozen(self.net.layers[:1])
        self.net.train()
        self._check_frozen(self.net.layers[:1])
        self._check_frozen(self.net.layers[1:], should_be_frozen=False)


class TestCNNEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.output_shape = torch.Size((16, 32))
        self.net = encoders.CNNEncoder(self.test_inputs.shape[1:], 3, 32)