from matplotlib import gridspec
from matplotlib import pyplot
import torch


def pit_hist(ax, x, n_bins, **kwargs):
    ax.stairs(x, torch.linspace(0, 1, n_bins + 1), **kwargs)


def get_grid():
    fig = pyplot.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    ax_true = fig.add_subplot(gs[1, 0])
    ax_pred = fig.add_subplot(gs[1, 1])
    return fig, ax, ax_true, ax_pred


def on_button_press(event, ax, model, plot_fn):
    x, y = float(event.xdata), float(event.ydata)
    if x is not None and y is not None:
        ax.clear()
        x_pred = model.decode(torch.tensor([[x, y]])).squeeze()
        plot_fn(ax, x_pred, label=f"({x:.1f}, {y:.1f})")
        ax.legend()
        fig.canvas.draw()


def on_pick(event, ax, dataset, model, plot_fn):
    idx = event.ind[0]
    ax.clear()
    # true
    x, y = dataset.X[idx], dataset.y[idx]
    plot_fn(ax, x, label="\n".join(map(repr, y)))
    # reconstruction
    mu, _ = model.encode(x.unsqueeze(0))
    plot_fn(ax, model.decode(mu).squeeze())
    ax.legend()
    fig.canvas.draw()
