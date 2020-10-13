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
