import pytorch_lightning as pl
import torchvision

import data
import lightning.modules
from models import encoders, decoders, bottlenecks


def run():
    pl.seed_everything(42)

    datamodule = data.MNISTDataModule('../data')

    encoder = encoders.DenseEncoder(datamodule.dims, 3, 32)
    decoder = decoders.DenseDecoder(32, 3, datamodule.dims)
    bottleneck = bottlenecks.IdentityBottleneck()
    ae = lightning.modules.Autoencoder(encoder, bottleneck, decoder)

    trainer = pl.Trainer(max_epochs=50, deterministic=True)
    trainer.fit(ae, datamodule=datamodule)


if __name__ == '__main__':
    run()
