import pytorch_lightning as pl

import data
from building import build_ae, build_datamodule, build_logger

AUTOENCODERS = ['shallow',
                'vanilla',
                'stacked',
                'sparse',
                'denoising',
                'vae',
                'beta_vae_strict',
                'beta_vae_loose',
                'vq']


def run(model_type, dataset, batch_size, gpu, anomaly=False):
    assert model_type in AUTOENCODERS
    task = 'anomaly' if anomaly else None
    pl.seed_everything(42)
    datamodule = build_datamodule(dataset, model_type, batch_size, anomaly)
    ae = build_ae(model_type, datamodule.dims, anomaly)
    logger = build_logger(model_type, dataset, task)
    checkpoint_path = _train(model_type, ae, datamodule, logger, gpu)

    return checkpoint_path


def _train(model_type, ae, datamodule, logger, gpu):
    epochs = 60
    gpus = [0] if gpu else None
    if model_type == 'stacked':
        trainer = _train_stacked(ae, datamodule, logger, epochs, gpus)
    else:
        trainer = _train_normal(ae, datamodule, logger, epochs, gpus)
    checkpoint_path = trainer.checkpoint_callback.last_model_path

    return checkpoint_path


def _train_stacked(ae, datamodule, logger, epochs, gpus):
    epochs_per_layer = _get_epochs_per_layer(epochs, ae.encoder.num_layers)
    trainer = pl.Trainer(max_epochs=0, deterministic=True, logger=logger, gpus=gpus)
    for additional_epochs in epochs_per_layer:
        trainer.max_epochs += additional_epochs
        trainer.fit(ae, datamodule=datamodule)
        trainer.current_epoch += 1
        ae.encoder.stack_layer()
        ae.decoder.stack_layer()

    return trainer


def _get_epochs_per_layer(epochs, num_layers):
    epochs_per_layer = num_layers * [epochs // num_layers]
    epochs_per_layer[-1] += epochs % num_layers

    return epochs_per_layer


def _train_normal(ae, datamodule, logger, epochs, gpus):
    trainer = pl.Trainer(max_epochs=epochs, deterministic=True, logger=logger, gpus=gpus)
    trainer.fit(ae, datamodule=datamodule)

    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run unsupervised autoencoder training.')
    parser.add_argument('model_type', choices=AUTOENCODERS)
    parser.add_argument('--dataset', default='mnist', choices=data.AVAILABLE_DATASETS.keys())
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--gpu', action='store_true', help='use GPU for training')
    parser.add_argument('--anomaly', action='store_true', help='train for anomaly detection')
    opt = parser.parse_args()

    print(run(opt.model_type, opt.dataset, opt.batch_size, opt.gpu, opt.anomaly))
