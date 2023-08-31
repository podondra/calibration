import torch


START = -5
STEPS = 100


def pdf(ax, dist, start=START, end=-START, steps=STEPS, **kwargs):
    x = torch.linspace(start, end, steps)
    ax.plot(x, dist.pdf(x), **kwargs)


def dists(ax, dist_pred, dist_data, start=START, end=-START, steps=STEPS):
    pdf(ax, dist_pred, start, end, steps, label="predictive")
    pdf(ax, dist_data, start, end, steps, label="data generating")
    ax.legend()


def pit_hist(ax, x, n_bins, **kwargs):
    ax.stairs(x, torch.linspace(0, 1, n_bins + 1), **kwargs)
