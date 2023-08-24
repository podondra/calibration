import itertools
import math

import torch

from . import dists


def pit(dist, y):
    return dist.cdf(y)


def pit_hist(x, n_bins):
    return torch.histc(x, bins=n_bins, min=0, max=1) / len(x)


def y2dists(w, d, s1, s2):
    dist_true = dists.Mixture(torch.stack((w, 1 - w)),
                              [dists.Normal(-d / 2, s1), dists.Normal(d / 2, s2)])
    return dists.Normal(0, 1), dist_true


class PITHistDataset(torch.utils.data.Dataset):
    def __init__(self, samples, n_bins, device=None):
        weights = torch.linspace(0.0, 1.0, steps=5, device=device)
        x = torch.linspace(0.1, 1.0, steps=5, device=device)
        separation = 2 * (1 - x ** 2)
        scales = torch.logspace(-1, 1, steps=5, base=2, device=device)
        # generate data
        self.n = len(separation) * len(weights) * len(scales) ** 2
        self.X = torch.empty(self.n, n_bins, device=device)
        self.y = torch.empty(self.n, 4, device=device)
        counter = itertools.count()
        dist_pred = dists.Normal(0, 1)
        for w in weights:
            for d in separation:
                for s1 in scales:
                    for s2 in scales:
                        ws = torch.stack((w, 1 - w))
                        mu = torch.stack((-d / 2, d / 2))
                        sigma = torch.stack((s1, s2))
                        sample = dists.sample_normal_mixture(ws, mu, sigma, samples, device)
                        pit_values = pit(dist_pred, sample)
                        i = next(counter)
                        self.X[i] = pit_hist(pit_values, n_bins)
                        self.y[i] = torch.stack((w, d, s1, s2))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class PITHistSampler(torch.utils.data.Dataset):
    def __init__(self, n, samples, n_bins, device=None):
        self.n = n
        self.samples = samples
        self.n_bins = n_bins
        self.device = device
        self.dist_pred = dists.Normal(torch.tensor(0.0), torch.tensor(1.0))

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        # weights
        w = torch.rand(torch.Size(), device=self.device)
        ws = torch.stack((w, 1 - w))
        # separtation implies means
        x = 0.1 + 0.9 * torch.rand(torch.Size(), device=self.device)
        d = 2.0 * (1 - x ** 2)
        mu = torch.stack((-d / 2, d / 2))
        # scales
        sigma = 2 ** (-1 + 2 * torch.rand(2, device=self.device))
        # generate
        sample = dists.sample_normal_mixture(ws, mu, sigma, self.samples, self.device)
        pit_values = pit(self.dist_pred, sample)
        return pit_hist(pit_values, self.n_bins), torch.stack((w, d, sigma[0], sigma[1]))


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.X = dataset.data.to(device=device, dtype=torch.float).flatten(1)
        self.X /= 256
        self.y = dataset.targets.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
