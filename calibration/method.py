import copy
import math

import torch
import wandb

from . import dist


def epochs(model, loader, trainset, validset, optimiser, hyperparams):
    for _ in range(hyperparams["epochs"]):
        model.train(loader, optimiser)
        log_train = trainset.evaluate(model)
        log_valid = validset.evaluate(model)
        wandb.log({"train": log_train, "valid": log_valid})


def early_stopping(model, loader, trainset, validset, optimiser, hyperparams):
    wandb.define_metric("valid.loss", summary="min")
    loss_best = float("inf")
    i = 0
    while i < hyperparams["patience"]:
        model.train(loader, optimiser)
        log_train = trainset.evaluate(model)
        log_valid = validset.evaluate(model)
        if log_valid["loss"] < loss_best:
            loss_best = log_valid["loss"]
            model_state_dict_best = copy.deepcopy(model.state_dict())
            i = 0
        else:
            i += 1
        wandb.log({"train": log_train, "valid": log_valid})
    model.load_state_dict(model_state_dict_best)


@torch.no_grad()
def predict(model, X):
    return model(X)


class MDN(torch.nn.Module):
    def __init__(self, inputs, neurons, components):
        super().__init__()
        self.components = components
        self.linear1 = torch.nn.Linear(inputs, neurons)
        self.linear2 = torch.nn.Linear(neurons, 3 * components)

    def forward(self, x):
        z = self.linear2(torch.tanh(self.linear1(x)))
        # mixing coefficients
        alpha = torch.softmax(z[..., :self.components], dim=-1)
        # means
        mu = z[..., self.components:-self.components]
        # variances
        sigma = torch.exp(z[..., -self.components:])
        return alpha, mu, sigma

    def train(self, loader, optimiser):
        for x, sample in loader:
            optimiser.zero_grad()
            y_pred = self(x)
            loss = dist.nll_gaussian_mixture(sample, *y_pred).mean()
            loss.backward()
            optimiser.step()


class DE(torch.nn.Module):
    def __init__(self, inputs, neurons, members):
        super().__init__()
        self.members = list()
        for i in range(members):
            # random initialisation for every member
            self.members.append(MDN(inputs, neurons, 1))
            self.add_module(f"member{i}", self.members[-1])

    def forward(self, x):
        output = [m(x) for m in self.members]
        _, mus, sigmas = tuple(zip(*output))
        mus, sigmas = torch.concat(mus, dim=-1), torch.concat(sigmas, dim=-1)
        mu = torch.mean(mus, dim=-1, keepdim=True)
        sigma = (torch.mean(sigmas + mus.square(), dim=-1, keepdim=True)
                - mu.square())
        return torch.ones_like(mu), mu, sigma

    def train(self, loader, optimiser):
        for member in self.members:
            for x, sample in loader:    # random shuffling for every member
                optimiser.zero_grad()
                _, mu, sigma = self(x)
                loss = dist.nll_gaussian(sample, mu, sigma).mean()
                loss.backward()
                optimiser.step()
