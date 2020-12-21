import atexit
import itertools
import shutil
import tempfile

import matplotlib.pyplot as plt
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


def tempdir():
    tempdir_path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, tempdir_path)

    return tempdir_path


def get_axes_grid(num_subplots, ncols, ax_size):
    axes = _get_axes(num_subplots, ncols, ax_size)
    _deactivate_unused_axes(axes, num_subplots)

    return axes


def _get_axes(num_subplots, ncols, ax_size):
    nrows = num_subplots // num_subplots
    nrows += 1 if nrows * ncols < num_subplots else 0
    figsize = (ax_size * ncols, ax_size * nrows)
    fig, axes = plt.subplots(nrows, ncols,
                             sharey='all',
                             sharex='all',
                             figsize=figsize)
    axes = axes.ravel()
    return axes


def _deactivate_unused_axes(axes, num_subplots):
    unused_axes = axes[num_subplots:]
    for unused_ax in unused_axes:
        unused_ax.set_axis_off()
