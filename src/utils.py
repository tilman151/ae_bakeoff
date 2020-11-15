import itertools

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
