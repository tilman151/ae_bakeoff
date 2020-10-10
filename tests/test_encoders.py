import unittest

import torch

from models import encoders
from tests.templates import ModelTestsMixin


class TestDenseEncoder(ModelTestsMixin, unittest.TestCase):
    def setUp(self):
        self.test_inputs = torch.randn(16, 1, 32, 32)
        self.output_shape = torch.Size((16, 32))
        self.net = encoders.DenseEncoder(self.test_inputs.shape[1:], 3, 32)
