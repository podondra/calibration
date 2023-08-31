import math

import torch


def sample_gaussian_mixture(weight, mean, variance, size, device=None):
    mean = mean.unsqueeze(-1)
    sd = torch.sqrt(variance).unsqueeze(-1)
    sample = mean + sd * torch.randn(len(weight), size, device=device)
    index = torch.multinomial(weight, num_samples=size, replacement=True)
    return torch.gather(sample, 0, index.unsqueeze(0)).squeeze()


def nll_gaussian_mixture(weight, mean, variance, y):
    weight = weight.unsqueeze(1)
    mean = mean.unsqueeze(1)
    variance = variance.unsqueeze(1)
    y = y.unsqueeze(-1)
    return -torch.logsumexp(torch.log(weight)
                            - 0.5 * torch.log(2.0 * math.pi * variance)
                            - 0.5 * ((y - mean) ** 2 / variance), dim=-1)


class Gaussian:
    def __init__(self, mean=torch.tensor(0.0), variance=torch.tensor(1.0)):
        self.mean = mean
        self.variance = variance
        self.sd = torch.sqrt(variance)

    def __repr__(self):
        return f"Gaussian({self.mean:.1f}, {self.variance:.1f})"

    def cdf(self, x):
        return 0.5 + 0.5 * torch.erf((x - self.mean)
                                     / (self.sd * math.sqrt(2)))

    def pdf(self, x):
        return ((1.0 / (self.sd * math.sqrt(2.0 * math.pi)))
                * torch.exp(-0.5 * ((x - self.mean) / self.sd) ** 2))

    def sample(self, size, device):
        return self.mean + self.sd * torch.randn(size, device=device)


class Mixture:
    def __init__(self, weight, dist):
        self.weight, self.dist = weight, dist

    def __repr__(self):
        return f"Mixture({self.weight}, {repr(self.dist)})"

    def pdf(self, x):
        return torch.stack([w * d.pdf(x)
                            for w, d in zip(self.weight, self.dist)]).sum(0)
