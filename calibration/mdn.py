import math

import torch

from . import dist


def params2dist(alpha, mu, sigma):
    alpha = alpha.squeeze()
    mu = mu.squeeze()
    sigma = sigma.squeeze()
    return dist.Mixture(alpha,
                        [dist.Gaussian(m, s) for m, s in zip(mu, sigma)])


class MDN(torch.nn.Module):
    def __init__(self, inputs, neurons, m):
        super().__init__()
        self.m = m
        self.linear1 = torch.nn.Linear(inputs, neurons)
        self.linear2 = torch.nn.Linear(neurons, 3 * m)

    def forward(self, x):
        z = self.linear2(torch.tanh(self.linear1(x)))
        alpha = torch.softmax(z[..., :self.m], dim=-1)    # mixing coefficients
        mu = z[..., self.m:-self.m]    # means
        sigma = torch.exp(z[..., -self.m:])    # variances
        return alpha, mu, sigma

    def train(self, loader, optimiser):
        for x, sample in loader:
            optimiser.zero_grad()
            alpha, mu, sigma = self(x)
            loss = dist.nll_gaussian_mixture(alpha, mu, sigma, sample).mean()
            loss.backward()
            optimiser.step()
        return {"loss": loss}

    @torch.no_grad()
    def evaluate(self, dataset):
        y_pred = self(dataset.histogram)
        y = dataset.sample
        return {"loss": dist.nll_gaussian_mixture(*y_pred, y).mean()}

    @torch.no_grad()
    def predict(self, X):
        return self(X)
