from functools import reduce
from math import pow

import torch.nn as nn


class DenseDecoder(nn.Module):
    def __init__(self, latent_dim, num_layers, output_shape):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_shape = output_shape

        self.layers = self._build_layers()

    def _build_layers(self):
        final_units = reduce(lambda a, b: a*b, self.output_shape)
        shrinkage = int(pow(final_units // self.latent_dim, 1 / self.num_layers))
        in_units = self.latent_dim
        out_units = in_units * shrinkage

        layers = []
        for i in range(self.num_layers - 1):
            layers += [nn.Linear(in_units, out_units, bias=False),
                       nn.BatchNorm1d(out_units),
                       nn.ReLU(True)]
            in_units = out_units
            out_units = out_units * shrinkage

        layers += [nn.Linear(in_units, final_units)]

        return nn.Sequential(*layers)

    def forward(self, inputs):
        outputs = self.layers(inputs)
        outputs = outputs.view(-1, *self.output_shape)

        return outputs
