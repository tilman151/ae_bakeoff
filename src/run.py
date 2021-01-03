import pytorch_lightning as pl

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


def run(model_type, anomaly=False):
    assert model_type in AUTOENCODERS
    task = 'anomaly' if anomaly else None
    pl.seed_everything(42)
    datamodule = build_datamodule(model_type, anomaly=anomaly)
    ae = build_ae(model_type, datamodule.dims)
    logger = build_logger(model_type, task)
    checkpoint_path = _train(model_type, ae, datamodule, logger)

    return checkpoint_path


def _train(model_type, ae, datamodule, logger):
    epochs = 60
    if model_type == 'stacked':
        trainer = _train_stacked(ae, datamodule, logger, epochs)
    else:
        trainer = _train_normal(ae, datamodule, logger, epochs)
    checkpoint_path = trainer.checkpoint_callback.last_model_path

    return checkpoint_path


def _train_stacked(ae, datamodule, logger, epochs):
    epochs_per_layer = _get_epochs_per_layer(epochs, ae.encoder.num_layers)
    trainer = pl.Trainer(max_epochs=0, deterministic=True, logger=logger)
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


def _train_normal(ae, datamodule, logger, epochs):
    trainer = pl.Trainer(max_epochs=epochs, deterministic=True, logger=logger)
    trainer.fit(ae, datamodule=datamodule)

    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run unsupervised autoencoder training.')
    parser.add_argument('model_type', choices=AUTOENCODERS)
    parser.add_argument('--anomaly', action='store_true', help='train for anomaly detection')
    opt = parser.parse_args()

    print(run(opt.model_type, opt.anomaly))
