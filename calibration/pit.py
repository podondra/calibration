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
    weight = torch.rand(torch.Size(), device=device)
    weights = torch.stack((weight, 1 - weight))
    # separation implies means
    a = 0.1 + 0.9 * torch.rand(torch.Size(), device=device)
    separation = 2.0 * (1 - a**2)
    mean = torch.stack((-separation / 2, separation / 2))
    variances = 2 ** (-2 + 4 * torch.rand(2, device=device))
    parameters = torch.stack((weight, separation, variances[0], variances[1]))
    outcomes = dist.sample_gaussian_mixture(weights, mean, variances, samples, device)
    histogram = pit_hist(
        pit_gaussian(
            outcomes,
            mean=torch.tensor(0.0, device=device),
            variance=torch.tensor(1.0, device=device),
        ),
        bins,
    )
    return parameters, outcomes, histogram


def evaluate(model, dataset):
    alpha, mu, sigma = method.predict(model, dataset.histograms)
    return {"loss": model.loss(dataset.parameters, alpha, mu, sigma).mean()}


class PITSampler(torch.utils.data.IterableDataset):
    def __init__(self, bs, bins, samples=SAMPLES, device=None):
        self.bs = bs
        self.samples = samples
        self.bins = bins
        self.device = device
        self.parameters = torch.empty(bs, 4, device=device)
        self.outcomes = torch.empty(bs, samples, device=device)
        self.histograms = torch.empty(bs, bins, device=device)

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.bs:
            raise StopIteration()
        parameters, outcomes, histogram = random_pit_hist(
            self.samples, self.bins, self.device
        )
        # save batch for evaluation
        self.parameters[self.i] = parameters
        self.outcomes[self.i] = outcomes
        self.histograms[self.i] = histogram
        self.i += 1
        return histogram, parameters

    def evaluate(self, model):
        return evaluate(model, self)


class PITDataset(torch.utils.data.Dataset):
    def __init__(self, n, bins=BINS, samples=SAMPLES, device=None):
        self.n = n
        self.parameters = torch.empty(self.n, 4, device=device)
        self.outcomes = torch.empty(self.n, samples, device=device)
        self.histograms = torch.empty(self.n, bins, device=device)
        for i in range(n):
            parameters, outcomes, histogram = random_pit_hist(samples, bins, device)
            self.histograms[i] = histogram
            self.outcomes[i] = outcomes
            self.parameters[i] = parameters

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.histograms[i], self.parameters[i]

    def evaluate(self, model):
        return evaluate(model, self)


class PITReference(torch.utils.data.Dataset):
    def __init__(self, bins=BINS, samples=SAMPLES, steps=5, device=None):
        zero = torch.tensor(0.0, device=device)
        one = torch.tensor(1.0, device=device)
        weights = torch.linspace(0.0, 1.0, steps, device=device)
        a = torch.linspace(0.1, 1.0, steps, device=device)
        separations = 2 * (1 - a**2)
        variances = torch.logspace(-2, 2, steps, base=2, device=device)
        # generate data
        self.n = len(separations) * len(weights) * len(variances) ** 2
        self.parameters = torch.empty(self.n, 4, device=device)
        self.outcomes = torch.empty(self.n, samples, device=device)
        self.histograms = torch.empty(self.n, bins, device=device)
        counter = itertools.count()
        for w in weights:
            for s in separations:
                for t in variances:
                    for v in variances:
                        i = next(counter)
                        weight = torch.stack((w, 1 - w))
                        mean = torch.stack((-s / 2, s / 2))
                        variance = torch.stack((t, v))
                        self.parameters[i] = torch.stack((w, s, t, v))
                        self.outcomes[i] = dist.sample_gaussian_mixture(
                            weight, mean, variance, samples, device
                        )
                        pit_values = pit_gaussian(self.outcomes[i], zero, one)
                        self.histograms[i] = pit_hist(pit_values, bins)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.histograms[i], self.parameters[i]
