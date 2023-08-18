import itertools
import math

import torch

from . import dists


def pit(dist, y):
    return dist.cdf(y)


def pit_hist(x, n_bins):
    return torch.histc(x, bins=n_bins, min=0, max=1) / len(x)


def label2dists(b, s, d, w):
    dist_pred = dists.Normal(b, s)
    sigma = 1 - d ** 2 / 4
    dist_true = dists.Mixture(torch.stack((w, 1 - w)),
                              [dists.Normal(-d / 2, sigma),
                               dists.Normal(d / 2, sigma)])
    return dist_pred, dist_true


class PITHistDataset(torch.utils.data.Dataset):
    def __init__(self, samples, n_bins, device):
        # under- and overestimation
        location = torch.tensor([-0.5, -0.25, 0.0, 0.25, 0.5], device=device)
        # under- and overdispersion
        scale = torch.tensor([0.9, 1.0, 1.1], device=device)
        # uni- and multimodal
        separation = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device)
        weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
        # generate data
        self.n = len(location) * len(scale) * len(separation) * len(weights)
        self.X = torch.empty(self.n, n_bins, device=device)
        self.y = torch.empty(self.n, 4, device=device)
        counter = itertools.count()
        for b in location:
            for s in scale:
                dist_pred = dists.Normal(b, s)
                for d in separation:
                    for w in weights:
                        ws = torch.stack((w, 1 - w))
                        mus = torch.stack((-d / 2, d / 2))
                        sigma = 1 - d ** 2 / 4
                        sample = dists.sample_normal_mixture(ws, mus, sigma,
                                                             samples, device)
                        pit_values = pit(dist_pred, sample)
                        i = next(counter)
                        self.X[i] = pit_hist(pit_values, n_bins)
                        self.y[i] = torch.stack((b, s, d, ws[0]))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class PITHistSampler(torch.utils.data.Dataset):
    def __init__(self, n, samples, n_bins, device):
        self.n = n
        self.samples = samples
        self.n_bins = n_bins
        self.device = device

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        # under- and overestimation
        b = -0.5 + torch.rand(torch.Size(), device=self.device)
        # under- and overdispersion
        s = 0.9 + 0.2 * torch.rand(torch.Size(), device=self.device)
        # uni- and multimodal
        d = 1.5 * torch.rand(torch.Size(), device=self.device)
        sigma = 1 - d ** 2 / 4
        w = torch.rand(torch.Size(), device=self.device)
        ws = torch.stack((w, 1 - w))
        mus = torch.stack((-d / 2, d / 2))
        sample = dists.sample_normal_mixture(ws, mus, sigma,
                                             self.samples, self.device)
        dist_pred = dists.Normal(b, s)
        pit_values = pit(dist_pred, sample)
        return pit_hist(pit_values, self.n_bins), torch.stack((b, s, d, w))


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, device):
        self.X = dataset.data.to(device=device, dtype=torch.float).flatten(1)
        self.X /= 256
        self.y = dataset.targets.to(device)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
