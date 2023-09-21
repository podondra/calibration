import itertools

import torch

from . import dist
from . import method


BINS = 20
SAMPLES = 10000


def pit_gaussian(x, mean=torch.tensor(0.0), variance=torch.tensor(1.0)):
    return dist.cdf_gaussian(x, mean, variance)


def pit_gaussian_mixture(x, weight, mean, variance):
    return torch.sum(weight * dist.cdf_gaussian(x, mean, variance), dim=-1)


def pit_hist(x, bins=BINS):
    return torch.histc(x, bins=bins, min=0, max=1) / (len(x) / bins)


def random_pit_hist(samples, bins=BINS, device=None):
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
    y = dist.sample_gaussian_mixture(weights, mean, variance, samples, device)
    X = pit_hist(pit_gaussian(y,
                              mean=torch.tensor(0.0, device=device),
                              variance=torch.tensor(1.0, device=device)), bins)
    annotation = torch.stack((weight, separation, variance[0], variance[1]))
    return X, y, annotation


def evaluate(dataset, model):
    alpha, mu, sigma = method.predict(model, dataset.X)
    return {"loss": dist.nll_gaussian_mixture(dataset.y,
                                              alpha, mu, sigma).mean()}


class PITSampler(torch.utils.data.IterableDataset):
    def __init__(self, bs, bins, samples=SAMPLES, device=None):
        self.bs = bs
        self.samples = samples
        self.bins = bins
        self.device = device
        self.X = torch.empty(bs, bins, device=device)
        self.y = torch.empty(bs, samples, device=device)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.bs:
            raise StopIteration()
        X, y, _ = random_pit_hist(self.samples, self.bins, self.device)
        self.X[self.i], self.y[self.i] = X, y   # save batch for evaluation
        self.i += 1
        return X, y

    def evaluate(self, model):
        return evaluate(self, model)


class PITDataset(torch.utils.data.Dataset):
    def __init__(self, n, bins=BINS, samples=SAMPLES, device=None):
        self.n = n
        self.X = torch.empty(self.n, bins, device=device)
        self.y = torch.empty(self.n, samples, device=device)
        self.annotation = torch.empty(self.n, 4, device=device)
        for i in range(n):
            X, y, annotation = random_pit_hist(samples, bins, device)
            self.X[i] = X
            self.y[i] = y
            self.annotation[i] = annotation

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def evaluate(self, model):
        return evaluate(self, model)


class PITReference(torch.utils.data.Dataset):
    def __init__(self, bins=BINS, samples=SAMPLES, steps=5, device=None):
        zero = torch.tensor(0.0, device=device)
        one = torch.tensor(1.0, device=device)
        weights = torch.linspace(0.0, 1.0, steps, device=device)
        x = torch.linspace(0.1, 1.0, steps, device=device)
        separations = 2 * (1 - x ** 2)
        variances = torch.logspace(-2, 2, steps, base=2, device=device)
        # generate data
        self.n = len(separations) * len(weights) * len(variances) ** 2
        self.y = torch.empty(self.n, samples, device=device)
        self.X = torch.empty(self.n, bins, device=device)
        self.annotation = torch.empty(self.n, 4, device=device)
        counter = itertools.count()
        for w in weights:
            for s in separations:
                for v1 in variances:
                    for v2 in variances:
                        i = next(counter)
                        weight = torch.stack((w, 1 - w))
                        mean = torch.stack((-s / 2, s / 2))
                        variance = torch.stack((v1, v2))
                        self.y[i] = dist.sample_gaussian_mixture(weight,
                                                                 mean,
                                                                 variance,
                                                                 samples,
                                                                 device)
                        pit_values = pit_gaussian(self.y[i], zero, one)
                        self.X[i] = pit_hist(pit_values, bins)
                        self.annotation[i] = torch.stack((w, s, v1, v2))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.y[i]
