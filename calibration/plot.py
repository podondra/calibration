from matplotlib import gridspec
from matplotlib import pyplot
import torch

from . import data


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


def get_grid(projection=None):
    fig = pyplot.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[:, 0], projection=projection)
    ax_true = fig.add_subplot(gs[0, 1])
    ax_pred = fig.add_subplot(gs[1, 1])
    return fig, ax, ax_true, ax_pred


def on_button_press(event, ax, model, plot_fn):
    x, y = float(event.xdata), float(event.ydata)
    if x is not None and y is not None:
        ax.clear()
        x_pred = model.decode(torch.tensor([[x, y]])).squeeze()
        ax.set_title(f"({x:.1f}, {y:.1f})")
        plot_fn(ax, x_pred)
        fig.canvas.draw()


def on_pick(event, ax, dataset, model, plot_fn):
    idx = event.ind[0]
    ax.clear()
    # true
    x, y = dataset.X[idx], dataset.y[idx]
    ax.set_title("\n".join(map(repr, data.label2dists(*y))))
    plot_fn(ax, x)
    # reconstruction
    mu, _ = model.encode(x.unsqueeze(0))
    plot_fn(ax, model.decode(mu).squeeze())
    fig.canvas.draw()
