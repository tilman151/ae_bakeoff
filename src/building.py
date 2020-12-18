import os

import pytorch_lightning.loggers as loggers

import data
import lightning
from models import encoders, decoders, bottlenecks


def build_datamodule(model_type=None, anomaly=False):
    apply_noise = (model_type == 'denoising')
    exclude = 9 if anomaly else None
    train_size = 550 if model_type == 'classification' else None
    datamodule = data.MNISTDataModule('../data',
                                      apply_noise=apply_noise,
                                      train_size=train_size,
                                      exclude=exclude)

    return datamodule


def build_ae(model_type, input_shape):
    latent_dim = 32
    encoder, decoder = _build_networks(model_type, input_shape, latent_dim)
    bottleneck = _build_bottleneck(model_type, latent_dim)
    ae = lightning.Autoencoder(encoder, bottleneck, decoder, lr=0.001)

    return ae


def _build_networks(model_type, input_shape, latent_dim):
    enc_dim = dec_dim = latent_dim
    if model_type == 'vae' or model_type.startswith('beta_vae'):
        enc_dim *= 2

    num_layers = 3
    if model_type == 'shallow':
        encoder = encoders.ShallowEncoder(input_shape, enc_dim)
        decoder = decoders.ShallowDecoder(dec_dim, input_shape)
    elif model_type == 'stacked':
        encoder = encoders.StackedEncoder(input_shape, num_layers, enc_dim)
        decoder = decoders.StackedDecoder(dec_dim, num_layers, input_shape)
    else:
        encoder = encoders.DenseEncoder(input_shape, num_layers, enc_dim)
        decoder = decoders.DenseDecoder(dec_dim, num_layers, input_shape)

    return encoder, decoder


def _build_bottleneck(model_type, latent_dim):
    if model_type == 'vanilla' or model_type == 'stacked' or model_type == 'denoising':
        bottleneck = bottlenecks.IdentityBottleneck(latent_dim)
    elif model_type == 'vae':
        bottleneck = bottlenecks.VariationalBottleneck(latent_dim)
    elif model_type == 'beta_vae_strict':
        bottleneck = bottlenecks.VariationalBottleneck(latent_dim, beta=2.)
    elif model_type == 'beta_vae_loose':
        bottleneck = bottlenecks.VariationalBottleneck(latent_dim, beta=0.5)
    elif model_type == 'sparse':
        bottleneck = bottlenecks.SparseBottleneck(latent_dim, sparsity=0.1)
    elif model_type == 'vq':
        bottleneck = bottlenecks.VectorQuantizedBottleneck(latent_dim, num_categories=512)
    else:
        raise ValueError(f'Unknown model type {model_type}.')

    return bottleneck


def build_logger(model_type, task='general'):
    log_dir = _get_log_dir()
    experiment_name = f'{model_type}_{task}'
    logger = loggers.TensorBoardLogger(log_dir, experiment_name)

    return logger


def _get_log_dir():
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, '..', 'logs'))

    return log_dir
