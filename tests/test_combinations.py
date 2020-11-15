import unittest

import torch

from models import encoders, decoders, bottlenecks
from downstream.classification import Classifier
from tests.templates import ModelTestsMixin, FrozenLayerCheckMixin


class TestStackedAutoencoder(unittest.TestCase):
    def test_gradients(self):
        enc = encoders.StackedEncoder((512,), 3, 32)
        dec = decoders.StackedDecoder(32, 3, (512,))
        optim = torch.optim.SGD(list(enc.parameters()) + list(dec.parameters()),
                                lr=0.01)

        inputs = torch.randn(16, 512)
        for i in range(3):
            with self.subTest(stack_level=i):
                optim.zero_grad()
                outputs = dec(enc(inputs))
                loss = torch.nn.functional.mse_loss(outputs, inputs)
                loss.backward()
                self._check_gradients(enc)
                self._check_gradients(dec)
                optim.step()
            enc.stack_layer()
            dec.stack_layer()

    def _check_gradients(self, net):
        for i, layer in enumerate(net.layers, start=1):
            if isinstance(layer, torch.nn.Sequential):
                layer = layer[0]
            if i == net.current_layer:
                self.assertIsNotNone(layer.weight.grad)
                self.assertNotEqual(0., layer.weight.grad.detach().sum())
            elif layer.weight.grad is not None:
                self.assertEqual(0., layer.weight.grad.detach().sum())
            else:
                self.assertIsNone(layer.weight.grad)


class TestClassification(ModelTestsMixin, FrozenLayerCheckMixin, unittest.TestCase):
    def setUp(self):
        encoder = encoders.DenseEncoder((512,), 3, 64)
        bottleneck = bottlenecks.VariationalBottleneck()
        self.net = Classifier(encoder, bottleneck, 32, 10)
        self.test_inputs = torch.randn(16, 512)
        self.output_shape = torch.Size((16, 10))

    def test_accuracy(self):
        accuracy = self.net._get_accuracy((self.test_inputs, torch.zeros(self.test_inputs.shape[0])))
        self.assertLessEqual(0, accuracy)
        self.assertGreaterEqual(1, accuracy)

    def test_layers_frozen(self):
        self._check_frozen(self.net.encoder)
