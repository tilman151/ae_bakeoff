import unittest

import torch

from models import decoders
from tests.templates import ModelTestsMixin


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


class TestStackedDecoder(unittest.TestCase):
    def setUp(self):
        self.output_shape = torch.Size((16, 1, 32, 32))
        self.net = decoders.StackedDecoder(32, 3, self.output_shape[1:])

    def test_stacking(self):
        for i, num_features in enumerate([341, 113, 32]):
            with self.subTest(stack_level=i):
                inputs = torch.randn(16, num_features)
                outputs = self.net(inputs)
                self.assertEqual(self.output_shape, outputs.shape)
            if self.net.current_layer < self.net.num_layers:
                self.net.stack_layer()
            else:
                self.assertRaises(RuntimeError, self.net.stack_layer)
