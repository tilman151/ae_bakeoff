import unittest
from unittest import mock

import building
import data
from models import bottlenecks, encoders, decoders


class TestBuildingDataModule(unittest.TestCase):
    def test_anomaly(self):
        dm = building.build_datamodule('mnist', model_type='vae', anomaly=True)
        self.assertEqual(1, dm.exclude)

    def test_classification(self):
        dm = building.build_datamodule('mnist', model_type='classification')
        self.assertEqual(550, dm.train_size)  # 1% of training data

    def test_no_model_type(self):
        with self.subTest(case='default'):
            dm = building.build_datamodule()
            self.assertIsInstance(dm, data.MNISTDataModule)
            self.assertIsNone(dm.exclude)
            self.assertIsNone(dm.train_size)
            self.assertEqual(32, dm.batch_size)
        with self.subTest(case='anomaly'):
            dm = building.build_datamodule(anomaly=True)
            self.assertIsInstance(dm, data.MNISTDataModule)
            self.assertEqual(1, dm.exclude)
            self.assertIsNone(dm.train_size)
            self.assertEqual(32, dm.batch_size)

    def test_rest(self):
        rest = ['shallow',
                'vanilla',
                'stacked',
                'sparse',
                'vae',
                'beta_vae_strict',
                'beta_vae_loose',
                'vq']
        for model_type in rest:
            for dataset in data.AVAILABLE_DATASETS.keys():
                with self.subTest(model_type=model_type):
                    dm = building.build_datamodule(dataset, model_type)
                    self.assertIsInstance(dm, data.AVAILABLE_DATASETS[dataset])
                    self.assertIsNone(dm.exclude)


class TestBuildingAE(unittest.TestCase):
    def test_build_bottleneck(self):
        with self.subTest(model_type='vanilla'):
            neck = self._build_neck('vanilla')
            self.assertIsInstance(neck, bottlenecks.IdentityBottleneck)
        with self.subTest(model_type='stacked'):
            neck = self._build_neck('stacked')
            self.assertIsInstance(neck, bottlenecks.IdentityBottleneck)
        with self.subTest(model_type='denoising'):
            neck = self._build_neck('denoising')
            self.assertIsInstance(neck, bottlenecks.IdentityBottleneck)
        with self.subTest(model_type='vae'):
            neck = self._build_neck('vae')
            self.assertIsInstance(neck, bottlenecks.VariationalBottleneck)
            self.assertEqual(1., neck.beta)
        with self.subTest(model_type='beta_vae_strict'):
            neck = self._build_neck('beta_vae_strict')
            self.assertIsInstance(neck, bottlenecks.VariationalBottleneck)
            self.assertEqual(2., neck.beta)
        with self.subTest(model_type='beta_vae_loose'):
            neck = self._build_neck('beta_vae_loose')
            self.assertIsInstance(neck, bottlenecks.VariationalBottleneck)
            self.assertEqual(0.5, neck.beta)
        with self.subTest(model_type='sparse'):
            neck = self._build_neck('sparse')
            self.assertIsInstance(neck, bottlenecks.SparseBottleneck)
            self.assertEqual(0.25, neck.sparsity)
            self.assertEqual(1., neck.beta)
        with self.subTest(model_type='vq'):
            neck = self._build_neck('vq')
            self.assertIsInstance(neck, bottlenecks.VectorQuantizedBottleneck)
            self.assertEqual(32, neck.latent_dim)
            self.assertEqual(512, neck.num_categories)
            self.assertEqual(1., neck.beta)
        with self.subTest(model_type='bogus'), self.assertRaises(ValueError):
            self._build_neck('bogus')

    @staticmethod
    def _build_neck(model_type):
        return building._build_bottleneck(model_type, latent_dim=32)

    def test_build_networks(self):
        latent_dim = 32
        with self.subTest(model_type='stacked'):
            encoder, decoder = self._build_nets('stacked', latent_dim)
            self.assertIsInstance(encoder, encoders.StackedEncoder)
            self.assertIsInstance(decoder, decoders.StackedDecoder)
            self.assertEqual(latent_dim, encoder.latent_dim)
            self.assertEqual(latent_dim, decoder.latent_dim)
        with self.subTest(model_type='shallow'):
            encoder, decoder = self._build_nets('shallow', latent_dim)
            self.assertIsInstance(encoder, encoders.ShallowEncoder)
            self.assertIsInstance(decoder, decoders.ShallowDecoder)
            self.assertEqual(latent_dim, encoder.latent_dim)
            self.assertEqual(latent_dim, decoder.latent_dim)
        with self.subTest(model_type='vanilla'):
            encoder, decoder = self._build_nets('vanilla', latent_dim)
            self.assertIsInstance(encoder, encoders.DenseEncoder)
            self.assertIsInstance(decoder, decoders.DenseDecoder)
            self.assertEqual(latent_dim, encoder.latent_dim)
            self.assertEqual(latent_dim, decoder.latent_dim)
        with self.subTest(model_type='vae'):
            encoder, decoder = self._build_nets('vae', latent_dim)
            self.assertIsInstance(encoder, encoders.DenseEncoder)
            self.assertIsInstance(decoder, decoders.DenseDecoder)
            self.assertEqual(2 * latent_dim, encoder.latent_dim)
            self.assertEqual(latent_dim, decoder.latent_dim)
        with self.subTest(model_type='beta_vae_strict'):
            encoder, decoder = self._build_nets('beta_vae_strict', latent_dim)
            self.assertIsInstance(encoder, encoders.DenseEncoder)
            self.assertIsInstance(decoder, decoders.DenseDecoder)
            self.assertEqual(2 * latent_dim, encoder.latent_dim)
            self.assertEqual(latent_dim, decoder.latent_dim)

    @staticmethod
    def _build_nets(model_type, latent_dim):
        return building._build_networks(model_type, (1, 32, 32), latent_dim)

    def test_denoising_build(self):
        with self.subTest(model_type='denoising'):
            ae = building.build_ae('denoising', (1, 32, 32), anomaly=False)
            self.assertNotEqual(0., ae.noise_ratio)
        with self.subTest(model_type='not denoising'):
            ae = building.build_ae('vanilla', (1, 32, 32), anomaly=False)
            self.assertEqual(0., ae.noise_ratio)

    def test_anomaly_build(self):
        with self.subTest(anomaly=False):
            ae = building.build_ae('vanilla', (1, 32, 32), anomaly=False)
            self.assertEqual(20, ae.encoder.latent_dim)
            self.assertEqual(20, ae.bottleneck.latent_dim)
            self.assertEqual(20, ae.decoder.latent_dim)
        with self.subTest(anomaly=True):
            ae = building.build_ae('vanilla', (1, 32, 32), anomaly=True)
            self.assertEqual(2, ae.encoder.latent_dim)
            self.assertEqual(2, ae.bottleneck.latent_dim)
            self.assertEqual(2, ae.decoder.latent_dim)


class TestBuildingLogger(unittest.TestCase):
    @mock.patch('os.path.dirname', return_value='/foo')
    def test__get_log_dir(self, mock_dirname):
        expected_logdir = '/logs/mnist'
        actual_log_dir = building._get_log_dir('mnist')
        self.assertEqual(expected_logdir, actual_log_dir)

    @mock.patch('os.path.dirname', return_value='/foo')
    def test_build_logger(self, mock_dirname):
        model_type = 'vae'
        dataset = 'mnist'
        with self.subTest(case='without task'):
            logger = building.build_logger(model_type, dataset)
            self.assertTrue(logger.root_dir.endswith(f'{model_type}_general'))
        with self.subTest(case='with task'):
            task = 'anomaly'
            logger = building.build_logger(model_type, dataset, task)
            self.assertTrue(logger.root_dir.endswith(f'{model_type}_{task}'))
