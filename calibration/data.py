import itertools

import torch

from . import dist


def pit(dist, y):
    return dist.cdf(y)


def pit_hist(x, bins):
    return torch.histc(x, bins=bins, min=0, max=1) / len(x)


def y2dist(weight, separation, variance1, variance2):
    return dist.Mixture(torch.stack((weight, 1 - weight)),
                        [dist.Gaussian(-separation / 2, variance1),
                         dist.Gaussian(+separation / 2, variance2)])


def random_pit_hist(samples, bins, device=None):
    dist_pred = dist.Gaussian(torch.tensor(0.0, device=device),
                              torch.tensor(1.0, device=device))
    # weights
    weight = torch.rand(torch.Size(), device=device)
    weights = torch.stack((weight, 1 - weight))
    # separation implies means
    x = 0.1 + 0.9 * torch.rand(torch.Size(), device=device)
    separation = 2.0 * (1 - x ** 2)
    mean = torch.stack((-separation / 2, separation / 2))
    # variance
    variance = 2 ** (-2 + 4 * torch.rand(2, device=device))
    # generate
    sample = dist.sample_gaussian_mixture(weights, mean, variance, samples,
                                          device)
    histogram = pit_hist(pit(dist_pred, sample), bins)
    y = torch.stack((weight, separation, variance[0], variance[1]))
    return sample, histogram, y


class PITSampler(torch.utils.data.IterableDataset):
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
        sample, histogram, y = random_pit_hist(self.samples,
                                               self.bins,
                                               self.device)
        return histogram, sample


class PITDataset(torch.utils.data.Dataset):
    def __init__(self, n, bins, samples=10000, device=None):
        self.n = n
        self.histogram = torch.empty(self.n, bins, device=device)
        self.sample = torch.empty(self.n, samples, device=device)
        self.y = torch.empty(self.n, 4, device=device)
        for i in range(n):
            sample, histogram, y = random_pit_hist(samples, bins, device)
            self.histogram[i] = histogram
            self.sample[i] = sample
            self.y[i] = y

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.histogram[i], self.sample[i]


class PITReference(torch.utils.data.Dataset):
    def __init__(self, bins, samples=10000, steps=5, device=None):
        dist_pred = dist.Gaussian(torch.tensor(0.0, device=device),
                                  torch.tensor(1.0, device=device))
        weights = torch.linspace(0.0, 1.0, steps, device=device)
        x = torch.linspace(0.1, 1.0, steps, device=device)
        separations = 2 * (1 - x ** 2)
        variances = torch.logspace(-2, 2, steps, base=2, device=device)
        # generate data
        self.n = len(separations) * len(weights) * len(variances) ** 2
        self.sample = torch.empty(self.n, samples, device=device)
        self.histogram = torch.empty(self.n, bins, device=device)
        self.y = torch.empty(self.n, 4, device=device)
        counter = itertools.count()
        for w in weights:
            for s in separations:
                for variance1 in variances:
                    for variance2 in variances:
                        i = next(counter)
                        weight = torch.stack((w, 1 - w))
                        mean = torch.stack((-s / 2, s / 2))
                        variance = torch.stack((variance1, variance2))
                        self.sample[i] = dist.sample_gaussian_mixture(weight,
                                                                      mean,
                                                                      variance,
                                                                      samples,
                                                                      device)
                        pit_values = pit(dist_pred, self.sample[i])
                        self.histogram[i] = pit_hist(pit_values, bins)
                        self.y[i] = torch.stack((w, s, variance1, variance2))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.histogram[i], self.sample[i]
