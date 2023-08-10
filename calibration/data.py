import itertools
import math

import numpy
import torch

from . import dists


def pit(dist, y):
    return dist.cdf(y)


def pit_hist(x, n_bins):
    return torch.histc(x, bins=n_bins, min=0, max=1) / len(x)


class PITHistDataset(torch.utils.data.Dataset):
    def __init__(self, sample_size, n_bins):
        # under- and overestimation: equidistant
        location = torch.linspace(-2, 2, steps=5)
        # under- and overdispersion: logarithmic
        scale = torch.exp(torch.linspace(math.log(1 / 3), math.log(3), steps=5))
        # uni- and multimodal
        x = torch.tensor([0.1, 0.25, 0.5, 0.75, 1])
        separation = 2 * (1 - x ** 2)
        # generate data
        n = len(location) * len(scale) * len(separation)
        self.X, self.y = torch.empty(n, n_bins), []
        i = itertools.count()
        for b in location:
            for s in scale:
                for d in separation:
                    dist_pred = dists.Normal(b, s)
                    sigma = 1 - d ** 2 / 4
                    dist_true = dists.Mixture([0.5, 0.5], [dists.Normal(-d / 2, sigma), dists.Normal(d / 2, sigma)])
                    samples = dist_true.sample(sample_size)
                    pit_values = pit(dist_pred, samples)
                    self.X[next(i)] = pit_hist(pit_values, n_bins)
                    self.y.append((dist_pred, dist_true, sample_size))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i]


class PITHistSampler(torch.utils.data.Dataset):
    def __init__(self, length, sample_size, n_bins):
        self.length = length
        self.sample_size = sample_size
        self.n_bins = n_bins

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        b = numpy.random.uniform(-2, 2)
        s = numpy.exp(numpy.random.uniform(math.log(1 / 3), math.log(3)))
        x = numpy.random.uniform(0.1, 1)
        d = 2 * (1 - x ** 2)
        dist_pred = dists.Normal(b, s)
        sigma = 1 - d ** 2 / 4
        dist_true = dists.Mixture([0.5, 0.5], [dists.Normal(-d / 2, sigma), dists.Normal(d / 2, sigma)])
        samples = dist_true.sample(self.sample_size)
        pit_values = pit(dist_pred, samples)
        return pit_hist(pit_values, self.n_bins)
