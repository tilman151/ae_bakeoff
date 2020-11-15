import warnings
from functools import reduce
from math import pow

import torch
import torch.nn as nn

import utils
from utils import pairwise


class DenseEncoder(nn.Module):
    def __init__(self, input_shape, num_layers, latent_dim):
        super().__init__()

        self.input_shape = input_shape
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.layers = self._build_layers()

    def _build_layers(self):
        units = self._get_units()
        layers = []
        for in_units, out_units in pairwise(units):
            layers += [self._build_hidden_layer(in_units, out_units)]
        layers += [self._build_final_layer(units[-1])]

        return nn.Sequential(*layers)

    def _get_units(self):
        in_units = reduce(lambda a, b: a * b, self.input_shape)
        shrinkage = int(pow(in_units // self.latent_dim, 1 / self.num_layers))
        units = [in_units // (shrinkage ** i) for i in range(self.num_layers)]

        return units

    @staticmethod
    def _build_hidden_layer(in_units, out_units):
        return nn.Sequential(nn.Linear(in_units, out_units, bias=False),
                             nn.BatchNorm1d(out_units),
                             nn.ReLU(True))

    def _build_final_layer(self, in_units):
        return nn.Linear(in_units, self.latent_dim)

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        outputs = self.layers(inputs)

        return outputs


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


class StackedEncoder(DenseEncoder):
    def __init__(self, input_shape, num_layers, latent_dim):
        super().__init__(input_shape, num_layers, latent_dim)

        self._current_layer = 1

    @property
    def current_layer(self):
        return self._current_layer

    def stack_layer(self):
        if self._current_layer < self.num_layers:
            self._current_layer += 1
            self._freeze_layers()
        else:
            warnings.warn('Encoder is already fully stacked.')

    def _freeze_layers(self):
        cut_off = self._current_layer - 1
        for m in self.layers[:cut_off].modules():
            utils.freeze_layer(m)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_layers()

    def forward(self, inputs):
        inputs = torch.flatten(inputs, start_dim=1)
        for n in range(self._current_layer):
            inputs = self.layers[n](inputs)

        return inputs
