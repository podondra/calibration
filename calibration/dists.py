import math

import torch


class Distribution:
    pass


class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def __repr__(self):
        return f"Normal({self.mu:.1f}, {self.sigma:.1f})"

    def pdf(self, x):
        return ((1.0 / (self.sigma * math.sqrt(2.0 * math.pi)))
                * torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2))

    def cdf(self, x):
        return 0.5 + 0.5 * torch.erf((x - self.mu) / (self.sigma * math.sqrt(2.0)))

    def sample(self, size, device):
        return self.mu + self.sigma * torch.randn(size, device=device)


def sample_normal_mixture(weights, mus, sigmas, size, device=None):
    mus, sigmas = mus.unsqueeze(-1),  sigmas.unsqueeze(-1)
    sample = mus + sigmas * torch.randn(2, size, device=device)
    index = torch.multinomial(weights, num_samples=size, replacement=True)
    return torch.gather(sample, 0, index.unsqueeze(0)).squeeze()


class Mixture(Distribution):
    def __init__(self, weights, dists):
        self.weights, self.dists = weights, dists

    def __repr__(self):
        return f"Mixture({self.weights}, {repr(self.dists)})"

    def pdf(self, x):
        return torch.stack([w * d.pdf(x) for w, d in zip(self.weights, self.dists)]).sum(0)

    def cdf(self, x):
        return torch.stack([w * d.cdf(x) for w, d in zip(self.weights, self.dists)]).sum(0)
