import unittest
from unittest import mock

import torch

from lightning import Autoencoder, AddNoise
from models import encoders, decoders, bottlenecks


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


class TestDenoisingAutoencoder(unittest.TestCase):
    def setUp(self):
        self.enc = encoders.DenseEncoder((512,), 2, 32)
        self.dec = decoders.DenseDecoder(32, 2, (512,))
        self.neck = bottlenecks.IdentityBottleneck(32)
        self.test_batch = [torch.randn(10, 512), torch.ones(10)]

    def test_setting_noise_ratio(self):
        with self.subTest(noise_ratio='default'):
            ae = Autoencoder(self.enc, self.neck, self.dec)
            self.assertEqual(0, ae.noise_ratio)
            self.assertEqual(0, ae.add_noise.noise_ratio)
        with self.subTest(noise_ratio='None'):
            ae = Autoencoder(self.enc, self.neck, self.dec, noise_ratio=None)
            self.assertEqual(0, ae.noise_ratio)
            self.assertEqual(0, ae.add_noise.noise_ratio)
        with self.subTest(noise_ratio='0.1'):
            ae = Autoencoder(self.enc, self.neck, self.dec, noise_ratio=0.1)
            self.assertEqual(0.1, ae.noise_ratio)
            self.assertEqual(0.1, ae.add_noise.noise_ratio)

    def test_clean_data_used_in_loss(self):
        features, labels = self.test_batch
        with self.subTest(case='noisy'):
            ae = Autoencoder(self.enc, self.neck, self.dec, noise_ratio=0.1)
            mock_encoder, mock_decoder, mock_criterion, mock_noise = self._mock_ae(ae, noise=True)
            ae.training_step(self.test_batch, batch_idx=0)
            self.assert_called_with_tensors(mock_noise, features)
            self.assert_not_called_with_tensors(mock_encoder, features)
            self.assert_called_with_tensors(mock_criterion, mock_decoder.return_value, features)
        with self.subTest(case='not noisy'):
            ae = Autoencoder(self.enc, self.neck, self.dec)
            mock_encoder, mock_decoder, mock_criterion, mock_noise = self._mock_ae(ae, noise=False)
            ae.training_step(self.test_batch, batch_idx=0)
            self.assert_called_with_tensors(mock_noise, features)
            self.assert_called_with_tensors(mock_encoder, features)
            self.assert_called_with_tensors(mock_criterion, mock_decoder.return_value, features)

    def _mock_ae(self, ae, noise):
        ae.encoder.forward = mock.MagicMock('encoder', wraps=ae.encoder.forward)
        ae.criterion_recon.forward = mock.MagicMock('criterion', wraps=ae.criterion_recon.forward)
        ae.decoder.forward = mock.MagicMock('decoder', return_value=torch.rand(10, 512))
        ae.add_noise.forward = mock.MagicMock('add_noise', wraps=lambda x: x + 1 if noise else x)

        return ae.encoder.forward, ae.decoder.forward, ae.criterion_recon.forward, ae.add_noise.forward

    def assert_not_called_with_tensors(self, mock_obj, *args):
        mock_obj.assert_called()
        for expected_arg, actual_arg in zip(args, mock_obj.call_args_list[-1][0]):
            self.assertNotEqual(0., torch.sum((expected_arg - actual_arg) ** 2))

    def assert_called_with_tensors(self, mock_obj, *args):
        mock_obj.assert_called()
        for expected_arg, actual_arg in zip(args, mock_obj.call_args_list[-1][0]):
            self.assertEqual(0., torch.sum((expected_arg - actual_arg) ** 2))

    def test_add_noise(self):
        features, _ = self.test_batch
        with self.subTest(noise=True, training=True):
            noise = AddNoise(noise_ratio=0.1)
            output = noise(features)
            self.assertNotEqual(0., torch.sum((features - output) ** 2))
        with self.subTest(noise=True, training=False):
            noise.eval()
            output = noise(features)
            self.assertEqual(0., torch.sum((features - output) ** 2))
        with self.subTest(noise=False, training=True):
            noise = AddNoise(noise_ratio=0)
            output = noise(features)
            self.assertEqual(0., torch.sum((features - output) ** 2))
        with self.subTest(noise=False, training=False):
            noise.eval()
            output = noise(features)
            self.assertEqual(0., torch.sum((features - output) ** 2))
