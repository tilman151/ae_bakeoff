import unittest

import numpy as np
import scipy.stats
import torch

from models import bottlenecks


class TestIdentityBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.IdentityBottleneck(32)

    def test_forward(self):
        inputs = torch.randn(16, 32)
        outputs, loss = self.neck(inputs)

        self.assertIs(inputs, outputs)
        self.assertEqual(0., loss)

    def test_sample(self):
        self.assertIsNone(self.neck.sample(1))  # Cannot sample from this bottleneck


class TestVariationalBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.VariationalBottleneck(1)

    @torch.no_grad()
    def test_forward(self):
        inputs = torch.zeros(16, 2)
        outputs, loss = self.neck(inputs)

        self.assertEqual(torch.Size((16, 1)), outputs.shape)

    @torch.no_grad()
    def test_kl_divergence(self):
        mu = np.random.randn(10) * 0.25
        sigma = np.random.randn(10) * 0.1 + 1.
        standard_normal_samples = np.random.randn(100000, 10)
        transformed_normal_sample = standard_normal_samples * sigma + mu

        # Calculate empirical pdfs for both distributions
        bins = 1000
        bin_range = [-2, 2]
        expected_kl_div = 0
        for i in range(10):
            standard_normal_dist, _ = np.histogram(standard_normal_samples[:, i], bins, bin_range)
            transformed_normal_dist, _ = np.histogram(transformed_normal_sample[:, i], bins, bin_range)
            expected_kl_div += scipy.stats.entropy(transformed_normal_dist, standard_normal_dist)
        expected_kl_div /= 10  # Account for batch_size

        inputs = torch.stack([torch.tensor(mu), torch.tensor(sigma).log()], dim=1)
        _, actual_kl_div = self.neck(inputs)

        self.assertAlmostEqual(expected_kl_div, actual_kl_div.numpy(), delta=0.05)

    def test_sample(self):
        sampled_latent = self.neck.sample(1000000)
        self.assertEqual(torch.Size((1000000, self.neck.latent_dim)), sampled_latent.shape)
        self.assertAlmostEqual(0, sampled_latent.numpy().mean(), places=2)
        self.assertAlmostEqual(1, sampled_latent.numpy().std(), places=2)


class TestSparseBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.SparseBottleneck(2, 0.5)

    @torch.no_grad()
    def test_forward(self):
        inputs = torch.zeros(16, 2)
        outputs, loss = self.neck(inputs)

        self.assertEqual(torch.Size((16, 2)), outputs.shape)

    @torch.no_grad()
    def test_kl_divergence(self):
        standard_samples = np.random.binomial(1, p=0.5, size=(1000000, 10))
        transformed_sample = np.random.binomial(1, p=0.2, size=(1000000, 10))

        # Calculate empirical pdfs for both distributions
        bins = 2
        bin_range = [0, 1]
        expected_kl_div = 0
        for i in range(10):
            standard_dist, _ = np.histogram(standard_samples[:, i], bins, bin_range)
            transformed_dist, _ = np.histogram(transformed_sample[:, i], bins, bin_range)
            expected_kl_div += scipy.stats.entropy(standard_dist, transformed_dist)

        inputs = (torch.tensor(transformed_sample, dtype=torch.float) - 0.5) * 1000
        _, actual_kl_div = self.neck(inputs)

        self.assertAlmostEqual(expected_kl_div, actual_kl_div.numpy(), delta=0.01)

    def test_sample(self):
        self.assertIsNone(self.neck.sample(1))  # Cannot sample from this bottleneck


class TestVectorQuantizedBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.VectorQuantizedBottleneck(16)

    def test_quantization(self):
        inputs = torch.randn(32, 16)
        actual_latent, _ = self.neck(inputs)
        expected_latent = self._vector_quantize_slow(inputs)

        self.assertEqual(inputs.shape, actual_latent.shape)
        self.assertTrue((expected_latent == actual_latent).all().item())

    def _vector_quantize_slow(self, inputs):
        dist = (self.neck.embeddings - inputs.unsqueeze(-1)) ** 2
        dist_idx = torch.argmin(dist, dim=-1)

        latent_code = torch.empty_like(inputs)
        for batch in range(latent_code.shape[0]):
            for dim in range(latent_code.shape[1]):
                latent_code[batch, dim] = self.neck.embeddings[0, dim, dist_idx[batch, dim]]

        return latent_code

    def test_loss(self):
        inputs = torch.randn(32, 16)
        latent, actual_loss = self.neck(inputs)
        expected_loss = torch.nn.functional.mse_loss(inputs, latent, reduction='sum')
        expected_loss += self.neck.beta * torch.nn.functional.mse_loss(latent, inputs, reduction='sum')
        expected_loss /= inputs.shape[0]

        self.assertAlmostEqual(expected_loss.item(), actual_loss.item(), places=5)

    def test_loss_gradient(self):
        inputs = torch.randn(32, 16)
        latent, loss = self.neck(inputs)
        loss.backward()

        self.assertFalse((self.neck.embeddings.grad == 0.).all())

    def test_straight_through_estimation(self):
        inputs = torch.randn(32, 16, requires_grad=True)
        latent, _ = self.neck(inputs)
        loss = latent.mean()
        loss.backward()

        with self.subTest(input_grad='has_grad'):
            self.assertFalse((inputs.grad == 0.).all())
        with self.subTest(embedding_grad='no_grad'):
            has_no_grad = (self.neck.embeddings.grad is None) or (self.neck.embeddings.grad == 0.).all()
            self.assertTrue(has_no_grad)

    def test_sample(self):
        sampled_latent = self.neck.sample(10)
        self.assertEqual(torch.Size((10, self.neck.latent_dim)), sampled_latent.shape)
        dist = (self.neck.embeddings - sampled_latent.unsqueeze(-1)) ** 2
        self.assertTrue(torch.all(torch.any(dist == 0, dim=2)))  # Each entry in sampled latent is in embedding
