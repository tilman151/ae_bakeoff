import itertools
import json
import os

import torch.nn as nn


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def freeze_layer(m):
    """Freezes the given layer for updates."""
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
    elif isinstance(m, nn.BatchNorm1d):
        if m.weight is not None:
            m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
        m.eval()


class ResultsMixin:
    def __init__(self):
        if self._results_exist():
            self.results = self._load_results()
        else:
            self.results = {}

    def __getitem__(self, key):
        return self.results[key]

    def __setitem__(self, key, value):
        self.results[key] = value

    def safe_add(self, model_type, key, value):
        if model_type in self.keys():
            self[model_type][key] = value
        else:
            self[model_type] = {key: value}

    def keys(self):
        return self.results.keys()

    def empty(self):
        return not self.results

    def missing_model_types(self, model_types):
        return list(set(model_types).difference(self.keys()))

    def save(self):
        checkpoint_path = self._get_results_path()
        with open(checkpoint_path, mode='wt') as f:
            json.dump(self.results, f, indent=4)

    def _results_exist(self):
        checkpoint_path = self._get_results_path()
        exists = os.path.exists(checkpoint_path)

        return exists

    def _load_results(self):
        checkpoint_path = self._get_results_path()
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, mode='rt') as f:
                checkpoints = json.load(f)
        else:
            raise FileNotFoundError(f'No checkpoint file found at {checkpoint_path}')

        return checkpoints

    def _get_results_path(self):
        raise NotImplementedError

    def _get_log_path(self):
        script_path = os.path.dirname(__file__)
        log_path = os.path.join(script_path, '..', 'logs')
        log_path = os.path.normpath(log_path)

        return log_path