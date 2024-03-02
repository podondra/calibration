import math

import torch


def nll_gaussian_mixture(x, weight, mean, variance):
    weight = weight.unsqueeze(1)
    mean = mean.unsqueeze(1)
    variance = variance.unsqueeze(1)
    x = x.unsqueeze(-1)
    return -torch.logsumexp(
        torch.log(weight)
        - 0.5 * torch.log(2.0 * math.pi * variance)
        - 0.5 * ((x - mean) ** 2 / variance),
        dim=-1,
    )


def nll_gaussian(x, mean, variance):
    return 0.5 * (torch.log(2 * math.pi * variance) + (x - mean) ** 2 / variance)


def cdf_gaussian(x, mean=torch.tensor(0.0), variance=torch.tensor(1.0)):
    return 0.5 + 0.5 * torch.erf((x - mean) / (torch.sqrt(2 * variance)))


def pdf_gaussian(x, mean=torch.tensor(0.0), variance=torch.tensor(1.0)):
    return (1.0 / (torch.sqrt(2 * math.pi * variance))) * torch.exp(
        -0.5 * ((x - mean) ** 2 / variance)
    )


def pdf_gaussian_mixture(x, weight, mean, variance):
    return torch.sum(weight * pdf_gaussian(x, mean, variance), dim=-1)


def crps_gaussian(x, mean, variance):
    # Gneiting et al. (2005, p. 1102)
    sd = torch.sqrt(variance)
    x_std = (x - mean) / sd
    return sd * (
        x_std * (2 * cdf_gaussian(x_std) - 1)
        + 2 * pdf_gaussian(x_std)
        - 1 / math.sqrt(math.pi)
    )


def A(mean, variance):
    # Grimit el al. (2006, p. 5)
    mean_std = mean / torch.sqrt(variance)
    return 2 * torch.sqrt(variance) * pdf_gaussian(mean_std) + mean * (
        2 * cdf_gaussian(mean_std) - 1
    )


def crps_gaussian_mixture(x, weight, mean, variance):
    # Grimit el al. (2006, p. 4)
    x = x.unsqueeze(-1)
    weight = weight.unsqueeze(1)
    mean = mean.unsqueeze(1)
    variance = variance.unsqueeze(1)
    crps = torch.sum(weight * A(x - mean, variance), dim=-1)
    crps -= 0.5 * torch.sum(
        weight * weight.mT * A(mean - mean.mT, variance + variance.mT), dim=(1, 2)
    ).unsqueeze(-1)
    return crps


def sample_gaussian_mixture(weight, mean, variance, size=10000, device=None):
    mean = mean.unsqueeze(-1)
    sd = torch.sqrt(variance).unsqueeze(-1)
    sample = mean + sd * torch.randn(len(weight), size, device=device)
    index = torch.multinomial(weight, num_samples=size, replacement=True)
    return torch.gather(sample, 0, index.unsqueeze(0)).squeeze()
