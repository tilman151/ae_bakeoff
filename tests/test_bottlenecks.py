import unittest

import torch
import numpy as np
import scipy.stats

from models import bottlenecks


class TestIdentityBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.IdentityBottleneck()

    def test_forward(self):
        inputs = torch.randn(16, 32)
        outputs, loss = self.neck(inputs)

        self.assertIs(inputs, outputs)
        self.assertEqual(0., loss)


class TestVariationalBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.VariationalBottleneck()

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

        inputs = torch.stack([torch.tensor(mu), torch.tensor(sigma).log()], dim=1)
        _, actual_kl_div = self.neck(inputs)

        self.assertAlmostEqual(expected_kl_div, actual_kl_div.numpy(), delta=0.05)


class TestSparseBottleneck(unittest.TestCase):
    def setUp(self):
        self.neck = bottlenecks.SparseBottleneck(0.5)

    @torch.no_grad()
    def test_forward(self):
        inputs = torch.zeros(16, 2)
        outputs, loss = self.neck(inputs)

        self.assertEqual(torch.Size((16, 1)), outputs.shape)

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
