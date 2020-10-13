from functools import reduce
from math import pow

import torch.nn as nn

from utils import pairwise


class DenseDecoder(nn.Module):
    def __init__(self, latent_dim, num_layers, output_shape):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_shape = output_shape

        self.layers = self._build_layers()

    def _build_layers(self):
        final_units = reduce(lambda a, b: a * b, self.output_shape)
        shrinkage = int(pow(final_units // self.latent_dim, 1 / self.num_layers))
        units = [final_units // (shrinkage ** i) for i in range(self.num_layers)]
        units.reverse()
        units = [self.latent_dim] + units

        layers = []
        for in_units, out_units in pairwise(units[:-1]):
            layers += [nn.Linear(in_units, out_units, bias=False),
                       nn.BatchNorm1d(out_units),
                       nn.ReLU(True)]

        layers += [nn.Linear(units[-2], units[-1]),
                   nn.Sigmoid()]

        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        outputs = outputs.view(-1, *self.output_shape)

        return outputs


class ShallowDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()

        self.output_shape = output_shape
        self.latent_dim = latent_dim

        out_units = reduce(lambda a, b: a * b, self.output_shape)
        self.layer = nn.Sequential(nn.Linear(self.latent_dim, out_units),
                                   nn.Sigmoid())

    def forward(self, inputs):
        outputs = self.layer(inputs)
        outputs = outputs.view(-1, *self.output_shape)

        return outputs
