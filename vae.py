from copy import deepcopy
from itertools import count
import math

import click
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import wandb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELPATH = "models/{}.pt"


def pit(dist, y):
    return dist.cdf(y)


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


class Distribution:
    pass


class Normal(Distribution):
    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def __eq__(self, other):
        if type(other) is not Normal:
            return False
        return self.mu == other.mu and self.sigma == other.sigma

    def __repr__(self):
        return f"Normal({self.mu:.1f}, {self.sigma:.1f})"

    def pdf(self, x):
        return ((1.0 / (self.sigma * torch.sqrt(2.0 * torch.tensor(math.pi))))
                * torch.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2))

    def cdf(self, x):
        return 0.5 + 0.5 * torch.erf((x - self.mu) / (self.sigma * math.sqrt(2.0)))

    def sample(self, size=1):
        size = size if type(size) is tuple else (size, )
        return torch.normal(self.mu, self.sigma, size)


class Mixture(Distribution):
    def __init__(self, weights, dists):
        assert math.isclose(sum(weights), 1.0)
        self.weights, self.dists = weights, dists

    def __eq__(self, other):
        if type(other) is not Mixture:   # simplification
            return False
        return self.weights == other.weights and self.dists == other.dists

    def __repr__(self):
        return f"Mixture({self.weights}, {repr(self.dists)})"

    def pdf(self, x):
        return torch.stack([w * d.pdf(x) for w, d in zip(self.weights, self.dists)]).sum(0)

    def cdf(self, x):
        return torch.stack([w * d.cdf(x) for w, d in zip(self.weights, self.dists)]).sum(0)

    def sample(self, size):
        js = np.random.choice(len(self.weights), p=self.weights, size=size)
        return to_tensor([self.dists[j].sample() for j in js.flatten()]).reshape(size)


class PITHistDataset(Dataset):
    def __init__(self, sample_size, n_bins):
        # under- and overestimation
        # equidistant
        location = torch.linspace(-2, 2, steps=5)
        # under- and overdispersion
        # logarithmic
        scale = torch.exp(torch.linspace(math.log(1 / 3), math.log(3), steps=5))
        # uni- and multimodal
        x = torch.tensor([0.1, 0.25, 0.5, 0.75, 1])
        separation = 2 * (1 - x ** 2)
        # generate data
        n = len(location) * len(scale) * len(separation)
        self.X, self.y = torch.empty(n, n_bins), []
        i = count()
        for b in location:
            for s in scale:
                for d in separation:
                    dist_pred = Normal(b, s)
                    sigma = 1 - d ** 2 / 4
                    dist_true = Mixture([0.5, 0.5], [Normal(-d / 2, sigma), Normal(d / 2, sigma)])
                    pit_values = pit(dist_pred, dist_true.sample(sample_size))
                    self.X[next(i)] = torch.histc(pit_values, bins=n_bins, min=0, max=1) / sample_size
                    self.y.append((dist_pred, dist_true, sample_size))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i]


class PITHistSampler(Dataset):
    def __init__(self, length, sample_size, n_bins):
        self.length = length
        self.sample_size = sample_size
        self.n_bins = n_bins

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        choice = np.random.uniform()
        if choice < 0.2:
            dist_true = Normal(0, 1)
            dist_pred = Normal(0, 1)
        elif choice > 0.5:
            w1 = np.clip(np.random.normal(0.5, 0.2), 0, 1)
            d = np.random.uniform(0, 2)
            s = np.exp(np.random.uniform(np.log(1 / 3), np.log(3)))
            dist_true = Mixture([w1, 1 - w1], [Normal(-d / 2, s), Normal(d / 2, s)])
            b = np.random.uniform(-2, 2)
            s = np.exp(np.random.uniform(np.log(1 / 3), np.log(3)))
            dist_pred = Normal(b, s)
        else:
            dist_true = Normal(0, 1)
            b = np.random.uniform(-2, 2)
            s = np.exp(np.random.uniform(np.log(1 / 3), np.log(3)))
            dist_pred = Normal(b, s)
        pit_values = pit(dist_pred, dist_true.sample(self.sample_size))
        return torch.histc(pit_values, bins=self.n_bins, min=0, max=1) / self.sample_size


def train_epochs(model, trainset, testset, hyperparams):
    optimiser = Adam(model.parameters(), lr=hyperparams["lr"])
    scheduler = StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
    loader = DataLoader(trainset, batch_size=hyperparams["bs"], num_workers=hyperparams["workers"])
    for epoch in range(1, hyperparams["epochs"] + 1):
        log_train = model.train(loader, optimiser)
        log_test = model.test(testset)
        wandb.log({"train": log_train, "test": log_test})
        scheduler.step()


def bhattacharyya(p_hist, q_hist):
    # TODO bug in pytorch? zeros in q_hist produces nans in gradients
    return -(p_hist * (q_hist + 1e-6)).sqrt().sum(1).log()


def mae(X_pred, X):
    return (X_pred - X).abs().mean(1)


def mse(X_pred, X):
    return (X_pred - X).square().mean(1)


def kl_divergence(mu, sigma):
    return 0.5 * (sigma.square() + mu.square() - 2 * sigma.log() - 1).sum(1)


class VAE(nn.Module):
    def __init__(self, input_dim, n_hiddens, n_neurons, embed_dim, epsilon):
        super().__init__()
        encoder = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hiddens - 1):
            encoder += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        encoder += [nn.Linear(n_neurons, embed_dim + 1)]
        self.encoder = nn.Sequential(*encoder).to(DEVICE)
        decoder = [nn.Linear(embed_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hiddens - 1):
            decoder += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        decoder += [nn.Linear(n_neurons, input_dim)]
        # TODO decoder += [nn.Softmax(dim=1)]
        self.decoder = nn.Sequential(*decoder).to(DEVICE)
        # hyperparams
        self.embed_dim = embed_dim
        self.epsilon = epsilon
        self.loss_rec = F.cross_entropy    # TODO

    def activation(self, x):
        return x[:, :self.embed_dim], torch.exp(x[:, self.embed_dim:])

    def forward(self, x):
        mu, sigma = self.activation(self.encoder(x))
        z = mu + sigma * torch.randn_like(mu)
        return self.decoder(z), mu, sigma

    @torch.no_grad()
    def encode(self, x):
        return self.activation(self.encoder(x.to(DEVICE)).cpu())

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x.to(DEVICE)).cpu().softmax(dim=1)    # TODO

    def train(self, loader, optimiser):
        for x in loader:
            optimiser.zero_grad()
            x = x.to(DEVICE)
            x_pred, mu, sigma = self(x)
            loss_rec = self.loss_rec(x_pred, x).mean()
            kl_div = kl_divergence(mu, sigma).mean()
            loss = (loss_rec + self.epsilon * kl_div)
            loss.backward()
            optimiser.step()
        return {"reconstruction": loss_rec.item(),
                "kl_divergence": kl_div.item(),
                "loss": loss.item()}

    @torch.no_grad()
    def test(self, dataset):
        X_pred, mu, sigma = self(dataset.X.to(DEVICE))
        X_pred, mu, sigma = X_pred.cpu(), mu.cpu(), sigma.cpu()
        loss_rec = self.loss_rec(X_pred, dataset.X).mean()
        kl_div = kl_divergence(mu, sigma).mean()
        loss = loss_rec + self.epsilon * kl_div
        return {"reconstruction": loss_rec.item(),
                "kl_divergence": kl_div.item(),
                "loss": loss.item()}


def seed():
    np.random.seed(18)
    torch.manual_seed(16)
    torch.backends.cudnn.benchmark = False


BINS = 10
SAMPLES = 1000


@click.command()
# data hyperparams
@click.option("--bins", default=BINS)
@click.option("--samples", default=SAMPLES)
# training hyperparams
@click.option("--bs", default=32)
@click.option("--epochs", default=1000)
@click.option("--gamma", default=1.0)
@click.option("--lr", default=1e-3)
@click.option("--step", default=1000)
@click.option("--workers", default=0)
# vae hyperparams
@click.option("--embed", default=2)
@click.option("--epsilon", default=1e-3)
@click.option("--hiddens", default=1)
@click.option("--neurons", default=8)
@click.option("--modelfile", type=click.Path(exists=True))
def vae(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        trainset = PITHistSampler(config["bs"], config["samples"], config["bins"])
        seed()    # reproducibility
        testset = PITHistDataset(config["samples"], config["bins"])
        model = VAE(
                config["bins"], config["hiddens"], config["neurons"], config["embed"],
                config["epsilon"])
        if config["modelfile"] is not None:
            model.load_state_dict(torch.load(config["modelfile"], map_location=DEVICE))
        train_epochs(model, trainset, testset, config)
        torch.save(model.state_dict(), MODELPATH.format(run.name))


if __name__ == "__main__":
    vae()
