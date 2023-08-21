import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb


def train_epochs(model, trainset, testset, optimiser, scheduler, hyperparams):
    loader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
    for epoch in range(1, hyperparams["epochs"] + 1):
        log_train = model.train(loader, optimiser)
        log_test = model.test(testset)
        wandb.log({"train": log_train, "test": log_test})
        scheduler.step()


def train_early_stopping(model, trainset, testset, optimiser, scheduler, hyperparams):
    loader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
    loss_rec_best = float("inf")
    i = 0
    while i < hyperparams["patience"]:
        log_train = model.train(loader, optimiser)
        log_test = model.test(testset)
        if log_test["reconstruction"] < loss_rec_best:
            loss_rec_best = log_test["reconstruction"]
            model_state_dict_best = copy.deepcopy(model.state_dict())
            i = 0
        else:
            i += 1
        wandb.log({"train": log_train, "test": log_test})
        wandb.run.summary["test.reconstruction"] = loss_rec_best
        scheduler.step()
    model.load_state_dict(model_state_dict_best)


def mse(a, b):
    return (a - b).square().mean(1)


def kl_divergence(mu, ln_var):
    return 0.5 * (ln_var.exp() + mu.square() - 1 - ln_var).sum(1)


class VAE(nn.Module):
    def __init__(self, inputs, hiddens, neurons, embeds, epsilon):
        super().__init__()
        encoder = [nn.Linear(inputs, neurons), nn.Tanh()]
        decoder = [nn.Linear(embeds, neurons), nn.Tanh()]
        for _ in range(hiddens - 1):
            encoder += [nn.Linear(neurons, neurons), nn.Tanh()]
            decoder += [nn.Linear(neurons, neurons), nn.Tanh()]
        decoder += [nn.Linear(neurons, inputs)]
        # TODO decoder += [nn.Softmax(dim=1)]
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(neurons, embeds)
        self.fc_ln_var = nn.Linear(neurons, embeds)
        self.decoder = nn.Sequential(*decoder)
        # hyperparams
        self.epsilon = epsilon
        self.loss_rec = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x):
        x = self.encoder(x)
        mu, ln_var = self.fc_mu(x), self.fc_ln_var(x)
        z = mu + (0.5 * ln_var).exp() * torch.randn_like(mu)
        return self.decoder(z), mu, ln_var

    @torch.no_grad()
    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), (0.5 * self.fc_ln_var(x)).exp()

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x).softmax(1)

    def train(self, loader, optimiser):
        loss_recs, kl_divs, losses, batches = 0, 0, 0, 0
        for x, _ in loader:
            optimiser.zero_grad()
            x_pred, mu, ln_var = self(x)
            loss_rec = self.loss_rec(x_pred, x).mean()
            kl_div = kl_divergence(mu, ln_var).mean()
            loss = loss_rec + self.epsilon * kl_div
            loss.backward()
            optimiser.step()
            loss_recs += loss_rec.item() * x.shape[0]
            kl_divs += kl_div.item() * x.shape[0]
            losses += loss.item() * x.shape[0]
            batches += x.shape[0]
        return {"reconstruction": loss_recs / batches,
                "kl_divergence": kl_divs / batches,
                "loss": losses / batches}

    @torch.no_grad()
    def test(self, dataset):
        X_pred, mu, ln_var = self(dataset.X)
        loss_rec = self.loss_rec(X_pred, dataset.X).mean()
        kl_div = kl_divergence(mu, ln_var).mean()
        loss = loss_rec + self.epsilon * kl_div
        log = {"reconstruction": loss_rec.item(),
               "kl_divergence": kl_div.item(),
               "loss": loss.item()}
        return log
