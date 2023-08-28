import math

import torch

from . import dists


def mdn2dist(alpha, mu, sigma):
    return dists.Mixture(alpha, [dists.Normal(m, s) for m, s in zip(mu, sigma)])


class MDN(torch.nn.Module):
    def __init__(self, inputs, neurons, hiddens, k):
        super().__init__()
        layers = [torch.nn.Linear(inputs, neurons), torch.nn.ReLU()]
        for _ in range(hiddens - 1):
            layers += [torch.nn.Linear(neurons, neurons), torch.nn.ReLU()]
        self.fc_hiddens = torch.nn.Sequential(*layers)
        self.fc_alpha = torch.nn.Linear(neurons, k)
        self.fc_mu = torch.nn.Linear(neurons, k)
        self.fc_ln_var = torch.nn.Linear(neurons, k)

    def forward(self, x):
        h = self.fc_hiddens(x)
        alpha = torch.softmax(self.fc_alpha(h), dim=-1)
        mu = self.fc_mu(h)
        ln_var = self.fc_ln_var(h)
        return alpha, mu, ln_var

    def criterion(self, alpha, mu, ln_var, y):
        alpha, mu, ln_var = alpha.unsqueeze(1), mu.unsqueeze(1), ln_var.unsqueeze(1)
        y = y.unsqueeze(-1)
        var = torch.exp(ln_var)
        return -torch.log(torch.sum(alpha
                                    * (1 / (torch.sqrt(2 * math.pi * var)))
                                    * torch.exp(-0.5 * ((y - mu) ** 2 / var)), dim=-1))
        # TODO torch.log(alpha) can be unstable, but it is softmax output
        #return -torch.logsumexp(torch.log(alpha)
        #                        - 0.5 * torch.log(2 * math.pi * var)
        #                        - 0.5 * ((y - mu) ** 2 / var), dim=-1)

    def train(self, loader, optimiser):
        for x, sample in loader:
            optimiser.zero_grad()
            alpha, mu, ln_var = self(x)
            loss = self.criterion(alpha, mu, ln_var, sample).mean()
            loss.backward()
            optimiser.step()
        return {"loss": loss.item()}

    @torch.no_grad()
    def evaluate(self, dataset):
        alpha, mu, ln_var = self(dataset.X)
        return {"loss": self.criterion(alpha, mu, ln_var, dataset.sample).mean().item()}

    @torch.no_grad()
    def predict(self, X):
        alpha, mu, ln_var = self(X)
        return alpha, mu, torch.exp(0.5 * ln_var)
