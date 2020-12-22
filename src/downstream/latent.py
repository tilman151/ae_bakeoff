import pytorch_lightning as pl
import torch
import torchvision
import umap

from building import load_ae_from_checkpoint


class Latent:
    def __init__(self, autoencoder):
        self.autoencoder = autoencoder.to('cpu')
        self.autoencoder.eval()

    @torch.no_grad()
    def sample(self, n):
        latent_code = self.autoencoder.bottleneck.sample(n)
        if latent_code is not None:
            samples = self.autoencoder.decoder(latent_code)
            samples = torchvision.utils.make_grid(samples).numpy()
        else:
            samples = None

        return samples

    def reconstruct(self, datamodule, num_comparison):
        recon_loss = self._get_reconstruction_loss(datamodule)
        comparison = self._build_reconstruction_comparison(datamodule, num_comparison)

        return recon_loss, comparison

    def _get_reconstruction_loss(self, datamodule):
        trainer = pl.Trainer(logger=False, deterministic=True)
        test_results, *_ = trainer.test(self.autoencoder, datamodule=datamodule)
        recon_loss = test_results['test/recon']

        return recon_loss

    @torch.no_grad()
    def _build_reconstruction_comparison(self, datamodule, n):
        batch = self._get_comparison_batch(datamodule, n)
        reconstruction = self.autoencoder(batch)
        comparison = self._build_comparison_grid(batch, reconstruction)

        return comparison

    def _get_comparison_batch(self, data, n):
        test_loader = data.test_dataloader()
        batch, _ = next(iter(test_loader))
        batch = batch[:n]

        return batch

    def _build_comparison_grid(self, batch, reconstruction):
        comparison = [tensor for sublist in zip(batch, reconstruction) for tensor in sublist]
        comparison = torchvision.utils.make_grid(comparison, nrow=2)
        comparison = comparison.numpy()

        return comparison

    @torch.no_grad()
    def interpolate(self, start, end, steps):
        steps += 2  # account for start only and end only steps
        start_encoded = self.autoencoder.encoder(start)
        end_encoded = self.autoencoder.encoder(end)
        interpolated_latents = [torch.lerp(start_encoded, end_encoded, w) for w in torch.linspace(0, 1, steps=steps)]
        interpolated_latents = [self.autoencoder.bottleneck(x)[0] for x in interpolated_latents]
        interpolated_samples = [self.autoencoder.decoder(inter) for inter in interpolated_latents]
        interpolated_samples = [torchvision.utils.make_grid(x, nrow=8) for x in interpolated_samples]
        interpolated_samples = torch.stack(interpolated_samples, dim=0).numpy()  # (steps x c x h x w)

        return interpolated_samples

    @torch.no_grad()
    def reduce(self, dataloader):
        labels = [label for _, label in dataloader]
        labels = torch.cat(labels)

        latents = [self.autoencoder.encoder(x) for x, _ in dataloader]
        latents = [self.autoencoder.bottleneck(x)[0] for x in latents]
        latents = torch.cat(latents).numpy()

        reduced_latents = umap.UMAP().fit_transform(latents)

        return reduced_latents, labels

    @classmethod
    def from_autoencoder_checkpoint(cls, model_type, dm, checkpoint_path):
        model = load_ae_from_checkpoint(model_type, dm.dims, checkpoint_path)
        latent = cls(model)

        return latent
