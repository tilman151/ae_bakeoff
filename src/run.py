import pytorch_lightning as pl

from building import build_ae, build_datamodule

autoencoders = ['shallow',
                'vanilla',
                'stacked',
                'sparse',
                'denoising',
                'vae',
                'beta_vae_strict',
                'beta_vae_loose',
                'vq']


def run(model_type):
    assert model_type in autoencoders
    pl.seed_everything(42)
    datamodule = build_datamodule(model_type)
    ae = build_ae(model_type, datamodule.dims)
    _train(model_type, ae, datamodule)


def _train(model_type, ae, datamodule):
    epochs = 60
    if model_type == 'stacked':
        _train_stacked(ae, datamodule, epochs)
    else:
        _train_normal(ae, datamodule, epochs)


def _train_stacked(ae, datamodule, epochs):
    num_stacking = ae.encoder.num_layers
    trainer = pl.Trainer(max_epochs=epochs // num_stacking, deterministic=True)
    for i in range(num_stacking):
        trainer.fit(ae, datamodule=datamodule)
        ae.encoder.stack_layer()
        ae.decoder.stack_layer()


def _train_normal(ae, datamodule, epochs):
    trainer = pl.Trainer(max_epochs=epochs, deterministic=True)
    trainer.fit(ae, datamodule=datamodule)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run unsupervised autoencoder training.')
    parser.add_argument('model_type', choices=autoencoders)
    opt = parser.parse_args()

    run(opt.model_type)
