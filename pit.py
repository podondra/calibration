from copy import deepcopy
import math

import click
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import wandb


BINS = 10
BS_TEST = 2048    # batch size for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELPATH = "models/{}.pt"
SAMPLES = 1000


def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def pit(dist, y):
    return dist.cdf(y)


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
        return f"Normal({self.mu}, {self.sigma})"

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


def generate_data(n_repeats, n_samples):
    samples, annotations = [], []
    locs = [0, 0.1, 0.5, 1, 3, -0.1, -0.5, -1, -3, 0, 0, 0, 0, 0, 0, 0, 0]
    scales = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1.1, 1.5, 3, 9, 0.9, 0.75, 0.4, 0.1]
    for repeat in range(n_repeats):
        # TODO refactor n_samples
        if n_repeats > 1: n_samples = int(100 ** (1 + repeat / (n_repeats - 1)))
        # unimodal
        dist_true = Normal(0, 1)
        for loc, scale in zip(locs, scales):
            dist_pred = Normal(loc, scale)
            pit_values = pit(dist_pred, dist_true.sample(n_samples))
            samples.append(pit_values)
            annotations.append((dist_pred, dist_true, n_samples))
        # multimodal
        for mu in [1, 2, 3]:
            dist_true = Mixture([0.5, 0.5], [Normal(-mu, 1), Normal(mu, 1)])
            for loc in [-3, -2, -1, 0, 1, 2, 3]:
                for scale in [1, 2, 3]:
                    dist_pred = Normal(loc, scale)
                    pit_values = pit(dist_pred, dist_true.sample(n_samples))
                    samples.append(pit_values)
                    annotations.append((dist_pred, dist_true, n_samples))
    return samples, annotations


def bin_data(pit_values, n_bins):
    return [torch.histc(p, bins=n_bins, min=0, max=1) / len(p) for p in pit_values]


class PITHistDataset(Dataset):
    def __init__(self, pit_values, annotations, n_bins):
        self.X, self.y = torch.stack(bin_data(pit_values, n_bins)), annotations

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.X[i]


def get_datasets(n_repeats, n_samples, n_bins):
    data_train = generate_data(n_repeats, n_samples)
    data_test = generate_data(1, n_samples)
    trainset = PITHistDataset(*data_train, n_bins)
    testset = PITHistDataset(*data_test, n_bins)
    return trainset, testset


@torch.no_grad()
def accuracy(z_train, y_train, z_test, y_test):
    knn = NearestNeighbors(n_neighbors=1).fit(z_train)
    js = knn.kneighbors(z_test, return_distance=False)
    n = len(y_test)
    return sum([y_test[i] == y_train[j[0]] for i, j in zip(range(n), js)]) / n


def train(model, loader, optimiser):
    for x, y in loader:
        optimiser.zero_grad()
        model.loss(model(x.to(DEVICE)), y.to(DEVICE)).mean().backward()
        optimiser.step()


def train_epochs(model, trainset, testset, hyperparams):
    optimiser = Adam(model.parameters(), lr=hyperparams["lr"])
    scheduler = StepLR(optimiser, step_size=hyperparams["step"], gamma=hyperparams["gamma"])
    loader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
    for epoch in range(1, hyperparams["epochs"] + 1):
        train(model, loader, optimiser)
        wandb.log(model.test(trainset, testset))
        scheduler.step()


def bhattacharyya(p_hist, q_hist):
    # TODO bug in pytorch? zeros in q_hist produces nans in gradients
    return -(p_hist * (q_hist + 1e-6)).sqrt().sum(1, keepdim=True).log()


def square_error(X_pred, X):
    return (X_pred - X).square().mean(1, keepdim=True)


def kl_divergence(mu, sigma):
    return 0.5 * (sigma.square() + mu.square() - 2 * sigma.log() - 1).sum(1, keepdim=True)


class Decoder(nn.Module):
    def __init__(self, n_hiddens, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(2, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, output_dim),
            nn.Softmax(dim=1)).to(DEVICE)

    def forward(self, x):
        return self.decoder(x)

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x.to(DEVICE)).cpu()


class VAE(nn.Module):
    def __init__(self, input_dim, n_hiddens, epsilon):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, 3)).to(DEVICE)
        self.decoder = Decoder(n_hiddens, input_dim)
        self.epsilon = epsilon
        self.loss_rec = square_error

    def activation(self, x):
        return x[:, :2], torch.exp(x[:, 2:])

    def forward(self, x):
        mu, sigma = self.activation(self.encoder(x))
        z = mu + sigma * torch.randn_like(mu)
        uni_pred = self.decoder(self.activation(self.encoder(torch.full((1, 10), 0.1, device=DEVICE)))[0])
        return self.decoder(z), uni_pred, mu, sigma

    @torch.no_grad()
    def embed(self, x):
        return self.activation(self.encoder(x.to(DEVICE)).cpu())

    def loss(self, X_pred, X):
        X_pred, uni_pred, mu, sigma = X_pred
        return self.loss_rec(X_pred, X) + self.loss_rec(torch.full((1, 10), 0.1, device=DEVICE), uni_pred) + self.epsilon * kl_divergence(mu, sigma)

    def test(self, trainset, testset):
        mu_train, sigma_train = self.embed(trainset.X)
        mu_test, sigma_test = self.embed(testset.X)
        z_train = mu_train + sigma_train * torch.randn_like(mu_train)
        z_test = mu_test + sigma_test * torch.randn_like(mu_test)
        X_pred_train = self.decoder.decode(z_train)
        X_pred_test = self.decoder.decode(z_test)
        return {
                "train": {
                    #"loss": self.loss((X_pred_train, mu_train, sigma_train), trainset.X).mean().item(),
                    "reconstruction": self.loss_rec(X_pred_train, trainset.X).mean().item(),
                    "kl_divergence": kl_divergence(mu_train, sigma_train).mean().item()},
                "test": {
                    #"loss": self.loss((X_pred_test, mu_test, sigma_test), testset.X).mean().item(),
                    "reconstruction": self.loss_rec(X_pred_test, testset.X).mean().item(),
                    "kl_divergence": kl_divergence(mu_test, sigma_test).mean().item(),
                    # TODO use probabilistic distance
                    "accuracy": accuracy(z_train, trainset.y, z_test, testset.y)}}


def seed():
    np.random.seed(18)
    torch.manual_seed(16)
    torch.backends.cudnn.benchmark = False


@click.group()
@click.option("--bs", default=32)
@click.option("--epochs", default=1000)
@click.option("--gamma", default=1.0)
@click.option("--lr", default=1e-3)
@click.option("--patience", default=10)
@click.option("--repeats", default=10)
@click.option("--samples", default=SAMPLES)
@click.option("--step", default=1000)
@click.pass_context
def experiment(context, **hyperparams):
    seed()   # reproducibility
    context.obj = hyperparams


@experiment.command()
@click.option("--bins", default=BINS)
@click.option("--epsilon", default=1e-3)
@click.option("--hiddens", default=8)
@click.pass_context
def vae(context, **hyperparams):
    hyperparams |= context.obj
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        trainset, testset = get_datasets(config["repeats"], config["samples"], config["bins"])
        model = VAE(config["bins"], config["hiddens"], config["epsilon"])
        train_epochs(model, trainset, testset, config)
        torch.save(model.state_dict(), MODELPATH.format(run.name))


if __name__ == "__main__":
    experiment(obj={})
