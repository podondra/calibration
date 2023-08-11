import itertools
import math

import torch

from . import dists


def pit(dist, y):
    return dist.cdf(y)


def pit_hist(x, n_bins):
    return torch.histc(x, bins=n_bins, min=0, max=1) / len(x)


class PITHistDataset(torch.utils.data.Dataset):
    def __init__(self, sample_size, n_bins, device):
        # under- and overestimation
        location = torch.linspace(-2, 2, steps=5)
        # under- and overdispersion
        scale = torch.exp(torch.linspace(math.log(1 / 3), math.log(3),
                                         steps=5))
        # uni- and multimodal
        x = torch.tensor([0.1, 0.25, 0.5, 0.75, 1])
        separation = 2 * (1 - x ** 2)
        # generate data
        self.n = len(location) * len(scale) * len(separation)
        self.X, self.y = torch.empty(self.n, n_bins), []
        i = itertools.count()
        for b in location:
            for s in scale:
                dist_pred = dists.Normal(b, s)
                for d in separation:
                    weights = torch.tensor([0.5, 0.5])
                    mus = torch.tensor([-d / 2, d / 2])
                    sigma = 1 - d ** 2 / 4
                    sample = dists.sample_normal_mixture(weights, mus, sigma,
                                                         sample_size)
                    pit_values = pit(dist_pred, sample)
                    self.X[next(i)] = pit_hist(pit_values, n_bins)
                    dist_true = dists.Mixture(weights,
                                              [dists.Normal(mus[0], sigma),
                                               dists.Normal(mus[1], sigma)])
                    self.y.append((dist_pred, dist_true, sample_size))
        self.X = self.X.to(device)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i]


class PITHistSampler(torch.utils.data.Dataset):
    def __init__(self, n, sample_size, n_bins, device):
        self.n = n
        self.sample_size = sample_size
        self.n_bins = n_bins
        self.device = device

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        # under- and overestimation from uniform [-2, 2)
        b = -2 + torch.rand(torch.Size(), device=self.device) * 4
        # under- and overdispersion: ln(3) = -1.0986
        s = torch.exp(-1.0986 + torch.rand(torch.Size(), device=self.device) * 2.1972)
        # uni- and multimodal
        x = 0.1 + torch.rand(torch.Size(), device=self.device) * 0.9
        d = 2 * (1 - x ** 2)
        sigma = 1 - d ** 2 / 4
        weight = torch.rand(torch.Size(), device=self.device)
        weights = torch.stack((weight, 1 - weight))
        mus = torch.stack((-d / 2, d / 2))
        sample = dists.sample_normal_mixture(weights, mus, sigma,
                                             self.sample_size, self.device)
        dist_pred = dists.Normal(b, s)
        pit_values = pit(dist_pred, sample)
        return pit_hist(pit_values, self.n_bins)
