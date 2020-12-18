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


def run(model_type, anomaly):
    assert model_type in AUTOENCODERS
    task = 'anomaly' if anomaly else None
    pl.seed_everything(42)
    datamodule = build_datamodule(model_type, anomaly)
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
    num_stacking = ae.encoder.num_layers
    trainer = pl.Trainer(max_epochs=epochs // num_stacking, deterministic=True, logger=logger)
    for i in range(num_stacking):
        trainer.fit(ae, datamodule=datamodule)
        ae.encoder.stack_layer()
        ae.decoder.stack_layer()

    return trainer


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
