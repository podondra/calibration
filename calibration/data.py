import itertools
import math

import torch

from . import dists


def pit(dist, y):
    return dist.cdf(y)


def pit_hist(x, bins):
    return torch.histc(x, bins=bins, min=0, max=1) / len(x)


def y2dist(w, d, s1, s2):
    return dists.Mixture(torch.stack((w, 1 - w)),
                         [dists.Normal(-d / 2, s1),
                          dists.Normal(d / 2, s2)])

def random_pit_hist(samples,
                    bins,
                    device=None):
    dist_pred = dists.Normal(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
    # weights
    w = torch.rand(torch.Size(), device=device)
    ws = torch.stack((w, 1 - w))
    # separtation implies means
    x = 0.1 + 0.9 * torch.rand(torch.Size(), device=device)
    d = 2.0 * (1 - x ** 2)
    mu = torch.stack((-d / 2, d / 2))
    # scales
    sigma = 2 ** (-1 + 2 * torch.rand(2, device=device))
    # generate
    sample = dists.sample_normal_mixture(ws, mu, sigma, samples, device)
    X = pit_hist(pit(dist_pred, sample), bins)
    y = torch.stack((w, d, sigma[0], sigma[1]))
    return sample, X, y


class PITHistSampler(torch.utils.data.IterableDataset):
    def __init__(self, bs, bins, samples=10000, device=None):
        self.bs = bs
        self.samples = samples
        self.bins = bins
        self.device = device

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        if self.i > self.bs:
            raise StopIteration()
        sample, X, y = random_pit_hist(self.samples, self.bins, device=self.device)
        return X, sample


class PITHistRndDataset(torch.utils.data.Dataset):
    def __init__(self, n, bins, samples=10000, device=None):
        self.n = n
        self.X = torch.empty(self.n, bins, device=device)
        self.sample = torch.empty(self.n, samples, device=device)
        self.y = torch.empty(self.n, 4, device=device)
        for i in range(n):
            sample, X, y = random_pit_hist(samples, bins, device=device)
            self.X[i] = X
            self.sample[i] = sample
            self.y[i] = y

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.sample[i]


class PITHistRefDataset(torch.utils.data.Dataset):
    def __init__(self, bins, samples=10000, steps=5, device=None):
        weights = torch.linspace(0.0, 1.0, steps, device=device)
        x = torch.linspace(0.1, 1.0, steps, device=device)
        separation = 2 * (1 - x ** 2)
        scales = torch.logspace(-1, 1, steps, base=2, device=device)
        # generate data
        self.n = len(separation) * len(weights) * len(scales) ** 2
        self.sample = torch.empty(self.n, samples, device=device)
        self.X = torch.empty(self.n, bins, device=device)
        self.y = torch.empty(self.n, 4, device=device)
        counter = itertools.count()
        dist_pred = dists.Normal(0, 1)
        for w in weights:
            for d in separation:
                for s1 in scales:
                    for s2 in scales:
                        i = next(counter)
                        ws = torch.stack((w, 1 - w))
                        mu = torch.stack((-d / 2, d / 2))
                        sigma = torch.stack((s1, s2))
                        self.sample[i] = dists.sample_normal_mixture(ws, mu, sigma, samples, device)
                        self.X[i] = pit_hist(pit(dist_pred, self.sample[i]), bins)
                        self.y[i] = torch.stack((w, d, s1, s2))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.sample[i]
