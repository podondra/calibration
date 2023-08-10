import math

import numpy
import torch


class Distribution:
    pass


class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def __repr__(self):
        return f"Normal({self.mu:.1f}, {self.sigma:.1f})"

    def pdf(self, x):
        return ((1.0 / (self.sigma * torch.sqrt(2.0 * torch.tensor(math.pi))))
                * torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2))

    def cdf(self, x):
        return 0.5 + 0.5 * torch.erf((x - self.mu) / (self.sigma * math.sqrt(2.0)))

    def sample(self, size=1):
        size = size if type(size) is tuple else (size, )
        return torch.normal(self.mu, self.sigma, size)


class Mixture(Distribution):
    def __init__(self, weights, dists):
        assert math.isclose(sum(weights), 1.0)
        self.weights, self.dists = weights, dists

    def __repr__(self):
        return f"Mixture({self.weights}, {repr(self.dists)})"

    def pdf(self, x):
        return torch.stack([w * d.pdf(x) for w, d in zip(self.weights, self.dists)]).sum(0)

    def cdf(self, x):
        return torch.stack([w * d.cdf(x) for w, d in zip(self.weights, self.dists)]).sum(0)

    def sample(self, size):
        js = numpy.random.choice(len(self.weights), p=self.weights, size=size)
        sample = [self.dists[j].sample() for j in js.flatten()]
        return torch.tensor(sample, dtype=torch.float32).reshape(size)
