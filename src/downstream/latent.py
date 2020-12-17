import torch
import torchvision
import umap


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

    @torch.no_grad()
    def reconstruct(self, batch):
        reconstruction = self.autoencoder(batch)
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
