from functools import reduce
from math import pow

import torch.nn as nn

from utils import pairwise


class DenseEncoder(nn.Module):
    def __init__(self, input_shape, num_layers, latent_dim):
        super().__init__()

        self.input_shape = input_shape
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.layers = self._build_layers()

    def _build_layers(self):
        if len(self.input_shape) > 1:
            layers = [nn.Flatten()]
        else:
            layers = []

        in_units = reduce(lambda a, b: a*b, self.input_shape)
        shrinkage = int(pow(in_units // self.latent_dim, 1 / self.num_layers))
        units = [in_units // (shrinkage ** i) for i in range(self.num_layers)]

        for in_units, out_units in pairwise(units):
            layers += [nn.Linear(in_units, out_units, bias=False),
                       nn.BatchNorm1d(out_units),
                       nn.ReLU(True)]

        layers += [nn.Linear(units[-1], self.latent_dim)]

        return nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class ShallowEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        in_units = reduce(lambda a, b: a * b, self.input_shape)
        self.layer = nn.Sequential(nn.Flatten(),
                                   nn.Linear(in_units, self.latent_dim),
                                   nn.ReLU())

    def forward(self, inputs):
        return self.layer(inputs)
