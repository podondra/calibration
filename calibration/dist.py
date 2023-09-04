import math

import torch


def nll_gaussian_mixture(x, weight, mean, variance):
    weight = weight.unsqueeze(1)
    mean = mean.unsqueeze(1)
    variance = variance.unsqueeze(1)
    x = x.unsqueeze(-1)
    return -torch.logsumexp(torch.log(weight)
                            - 0.5 * torch.log(2.0 * math.pi * variance)
                            - 0.5 * ((x - mean) ** 2 / variance), dim=-1)


# TODO implement directly
def nll_gaussian(x, mean, variance):
    return nll_gaussian_mixture(x, torch.ones_like(mean), mean, variance)


def cdf_gaussian(x, mean=0.0, variance=1.0):
    return 0.5 + 0.5 * torch.erf((x - mean)
                                 / (torch.sqrt(torch.tensor(2) * variance)))


def pdf_gaussian(x, mean=0.0, variance=1.0):
    return ((1.0 / (torch.sqrt(torch.tensor(2 * math.pi) * variance)))
            * torch.exp(-0.5 * ((x - mean) ** 2 / variance)))


def pdf_gaussian_mixture(x, weight, mean, variance):
    return torch.sum(weight * pdf_gaussian(x, mean, variance), dim=-1)


# TODO verify
def crps_gaussian(x, mean, variance):
    return torch.sqrt(variance) * (x * (2.0 * cdf_gaussian(x) - 1.0)
                                   + 2.0 * cdf_gaussian(x) - 1.0 / math.sqrt(math.pi))


# TODO rename
# TODO verify
def A(mean, variance):
    mean_std = mean / torch.sqrt(variance)
    return (2 * torch.sqrt(variance) * pdf_gaussian(mean_std)
            + mean * (2 * cdf_gaussian(mean_std) - 1))


# TODO verify
def crps_gaussian_mixture(x, weight, mean, variance):
    return torch.unsqueeze(torch.sum(weight * A(x - mean, variance), dim=1)
                           - 0.5
                           * torch.sum((weight[:, :, None] * weight[:, None, :])
                                       * A(mean[:, :, None] - mean[:, None, :],
                                           variance[:, :, None] + variance[:, None, :]),
                                       dim=(1, 2)),
                           dim=-1)


def sample_gaussian_mixture(weight, mean, variance, size=10000, device=None):
    mean = mean.unsqueeze(-1)
    sd = torch.sqrt(variance).unsqueeze(-1)
    sample = mean + sd * torch.randn(len(weight), size, device=device)
    index = torch.multinomial(weight, num_samples=size, replacement=True)
    return torch.gather(sample, 0, index.unsqueeze(0)).squeeze()
