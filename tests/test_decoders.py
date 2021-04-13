import unittest

import torch

from models import decoders
from templates import ModelTestsMixin, FrozenLayerCheckMixin


class DecoderTestsMixin(ModelTestsMixin):
    def test_multi_sample_shape(self):
        n_samples = 5
        test_inputs = self.test_inputs.unsqueeze(1).repeat(1, n_samples, *([1] * (self.test_inputs.ndim - 1)))
        output_shape = torch.Size((self.output_shape[0], n_samples, *self.output_shape[1:]))
        outputs = self.net(test_inputs)
        self.assertEqual(output_shape, outputs.shape)


class TestDenseEncoder(DecoderTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 32)
        self.output_shape = torch.Size((16, 1, 32, 32))
        self.net = decoders.DenseDecoder(32, 3, self.output_shape[1:])


class TestShallowDecoder(DecoderTestsMixin, unittest.TestCase):
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


class TestCNNDecoder(DecoderTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 32)
        self.output_shape = torch.Size((16, 1, 32, 32))
        self.net = decoders.CNNDecoder(32, 3, self.output_shape[1:])
