import warnings
from functools import reduce
from math import pow

import torch.nn as nn

import utils
from utils import pairwise


class DenseDecoder(nn.Module):
    def __init__(self, latent_dim, num_layers, output_shape):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.output_shape = output_shape

        self.layers = self._build_layers()

    def _build_layers(self):
        units = self._get_units()
        layers = []
        for in_units, out_units in pairwise(units[:-1]):
            layers += [self._build_hidden_layer(in_units, out_units)]
        layers += [self._build_final_layer(units[-2], units[-1])]

        return nn.Sequential(*layers)

    def _get_units(self):
        final_units = reduce(lambda a, b: a * b, self.output_shape)
        shrinkage = int(pow(final_units // self.latent_dim, 1 / self.num_layers))
        units = [final_units // (shrinkage ** i) for i in range(self.num_layers)]
        units.reverse()
        units = [self.latent_dim] + units

        return units

    @staticmethod
    def _build_hidden_layer(in_units, out_units):
        return nn.Sequential(nn.Linear(in_units, out_units, bias=False),
                             nn.BatchNorm1d(out_units),
                             nn.ReLU(True))

    @staticmethod
    def _build_final_layer(in_units, out_units):
        return nn.Sequential(nn.Linear(in_units, out_units),
                             nn.Sigmoid())

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


class StackedDecoder(DenseDecoder):
    def __init__(self, latent_dim, num_layers, output_shape):
        super().__init__(latent_dim, num_layers, output_shape)

        self._current_layer = self.num_layers

    @property
    def current_layer(self):
        return self._current_layer

    def stack_layer(self):
        if self._current_layer > 0:
            self._current_layer -= 1
            self._freeze_layers()
        else:
            warnings.warn('Decoder is already fully stacked.')

    def _freeze_layers(self):
        cut_off = self._current_layer
        for m in self.layers[cut_off:].modules():
            utils.freeze_layer(m)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_layers()

    def forward(self, inputs):
        for n in range(self._current_layer - 1, self.num_layers):
            inputs = self.layers[n](inputs)
        outputs = inputs.view(-1, *self.output_shape)

        return outputs
