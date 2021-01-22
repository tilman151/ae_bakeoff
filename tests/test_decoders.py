import unittest

import torch

from models import decoders
from templates import ModelTestsMixin, FrozenLayerCheckMixin


class TestDenseEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 32)
        self.output_shape = torch.Size((16, 1, 32, 32))
        self.net = decoders.DenseDecoder(32, 3, self.output_shape[1:])


class TestShallowDecoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 32)
        self.output_shape = torch.Size((16, 1, 32, 32))
        self.net = decoders.ShallowDecoder(32, self.output_shape[1:])


class TestStackedDecoder(unittest.TestCase, FrozenLayerCheckMixin):
    def setUp(self):
        self.output_shape = torch.Size((16, 1, 32, 32))
        self.net = decoders.StackedDecoder(32, 3, self.output_shape[1:])

    def test_stacking(self):
        for i, num_features in enumerate([341, 113, 32]):
            with self.subTest(stack_level=i):
                inputs = torch.randn(16, num_features)
                outputs = self.net(inputs)
                self.assertEqual(self.output_shape, outputs.shape)
            self.net.stack_layer()

    def test_freezing(self):
        for i in range(3, 0, -1):
            with self.subTest(stack_level=i):
                self._check_frozen(self.net.layers[i:])
                self._check_frozen(self.net.layers[:i], should_be_frozen=False)
            self.net.stack_layer()

    def test_set_training(self):
        self.net.stack_layer()
        self.net.eval()
        self._check_frozen(self.net.layers[2:])
        self.net.train()
        self._check_frozen(self.net.layers[2:])
        self._check_frozen(self.net.layers[:2], should_be_frozen=False)
