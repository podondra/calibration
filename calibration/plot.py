import torch

from . import pit


START = -5
STEPS = 100


def density(ax, pdf, start=START, end=-START, steps=STEPS, **kwargs):
    x = torch.linspace(start, end, steps)
    return ax.plot(x, pdf(x.unsqueeze(-1)), **kwargs)[0]


def pit_hist(ax, x, n_bins=pit.BINS, **kwargs):
    return ax.stairs(x, torch.linspace(0, 1, n_bins + 1), **kwargs)
