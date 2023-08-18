import copy

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb


def train_epochs(model, trainset, testset, hyperparams):
    optimiser = Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
    scheduler = StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
    loader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
    for epoch in range(1, hyperparams["epochs"] + 1):
        log_train = model.train(loader, optimiser)
        log_test = model.test(testset)
        wandb.log({"train": log_train, "test": log_test})
        scheduler.step()


def train_early_stopping(model, trainset, testset, hyperparams):
    optimiser = Adam(model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["wd"])
    scheduler = StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
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
    return (a - b).square().mean(1, keepdim=True)


def kl_divergence(mu, ln_var):
    return 0.5 * (ln_var.exp() + mu.square() - 1 - ln_var).sum(1, keepdim=True)


class AbstractVAE(nn.Module):
    def __init__(self, epsilon):
        super().__init__()
        # hyperparams
        self.epsilon = epsilon
        self.loss_rec = mse

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
        return self.decoder(x)

    def train(self, loader, optimiser):
        # TODO correct running loss log, but loader is always one batch only
        for x, _ in loader:
            optimiser.zero_grad()
            x_pred, mu, ln_var = self(x)
            loss_rec = self.loss_rec(x_pred, x).mean()
            kl_div = kl_divergence(mu, ln_var).mean()
            loss = loss_rec + self.epsilon * kl_div
            loss.backward()
            optimiser.step()
        return {"reconstruction": loss_rec.item(),
                "kl_divergence": kl_div.item(),
                "loss": loss.item()}

    @torch.no_grad()
    def test(self, dataset):
        X_pred, mu, ln_var = self(dataset.X)
        loss_rec = self.loss_rec(X_pred, dataset.X).mean()
        kl_div = kl_divergence(mu, ln_var).mean()
        loss = loss_rec + self.epsilon * kl_div
        log = {"reconstruction": loss_rec.item(),
               "kl_divergence": kl_div.item(),
               "loss": loss.item()}
        #log["embeddings"] = wandb.Table(columns=["x", "y"], data=mu.tolist())
        return log


class VAE(AbstractVAE):
    def __init__(self, inputs, hiddens, neurons, embeds, epsilon):
        super().__init__(epsilon)
        encoder = [nn.Linear(inputs, neurons), nn.Tanh()]
        decoder = [nn.Linear(embeds, neurons), nn.Tanh()]
        for _ in range(hiddens - 1):
            encoder += [nn.Linear(neurons, neurons), nn.Tanh()]
            decoder += [nn.Linear(neurons, neurons), nn.Tanh()]
        decoder += [nn.Linear(neurons, inputs)]
        decoder += [nn.Softmax(dim=1)]
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(neurons, embeds)
        self.fc_ln_var = nn.Linear(neurons, embeds)
        self.decoder = nn.Sequential(*decoder)


class ConvVAE(AbstractVAE):
    def __init__(self, inputs, embeds, epsilon):
        super().__init__(epsilon)
        kernel_size = 5
        padding = 2
        self.encoder = nn.Sequential(
                nn.Unflatten(1, (1, inputs)),
                nn.Conv1d(1, 4, kernel_size, padding=padding),
                nn.Tanh(),
                nn.Conv1d(4, 8, kernel_size, padding=padding),
                nn.Tanh(),
                nn.Flatten())
        self.fc_mu = nn.Linear(8 * inputs, embeds)
        self.fc_ln_var = nn.Linear(8 * inputs, embeds)
        self.decoder = nn.Sequential(
                nn.Linear(embeds, 8 * inputs), nn.Tanh(),
                nn.Unflatten(1, (8, inputs)),
                nn.ConvTranspose1d(8, 4, kernel_size, padding=padding),
                nn.Tanh(),
                nn.ConvTranspose1d(4, 1, kernel_size, padding=padding),
                nn.Flatten())
