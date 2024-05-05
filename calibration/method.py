import copy

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


def wasserstein_loss(parameters, alpha, mu, sigma):
    weight = torch.stack((parameters[:, 0], 1 - parameters[:, 0]), dim=-1)
    mean = torch.stack((-parameters[:, 1] / 2, parameters[:, 1] / 2), dim=-1)
    variance = parameters[:, 2:]
    return dist.one_wasserstein_distance(alpha, mu, sigma, weight, mean, variance)


class MDN(torch.nn.Module):
    def __init__(self, inputs, neurons, components, loss=dist.nll_gaussian_mixture):
        super().__init__()
        self.linear1 = torch.nn.Linear(inputs, neurons)
        self.linear2 = torch.nn.Linear(neurons, 3 * components)
        self.components = components
        self.loss = loss

    def forward(self, x):
        z = self.linear2(torch.tanh(self.linear1(x)))
        alpha = torch.softmax(z[..., : self.components], dim=-1)
        mu = z[..., self.components : -self.components]
        sigma = torch.exp(z[..., -self.components :])
        return alpha, mu, sigma

    def train(self, loader, optimiser):
        for x, y in loader:
            optimiser.zero_grad()
            y_pred = self(x)
            loss = self.loss(y, *y_pred).mean()
            loss.backward()
            optimiser.step()
