from copy import deepcopy
import math

import click
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import wandb


BS_TEST = 2048    # batch size for testing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELPATH = "models/{}.pt"


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
    # unimodal
    dist_true = Normal(0, 1)
    locs = [0, 0.1, 0.5, 1, 3, -0.1, -0.5, -1, -3, 0, 0, 0, 0, 0, 0, 0, 0]
    scales = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1.1, 1.5, 3, 9, 0.9, 0.75, 0.4, 0.1]
    for _ in range(n_repeats):
        for loc, scale in zip(locs, scales):
            dist_pred = Normal(loc, scale)
            pit_values = pit(dist_pred, dist_true.sample(n_samples))
            samples.append(pit_values)
            annotations.append((dist_pred, dist_true))
    # multimodal
    for _ in range(n_repeats):
        for mu in [1, 2, 3]:
            dist_true = Mixture([0.5, 0.5], [Normal(-mu, 1), Normal(mu, 1)])
            for loc in [-3, -2, -1, 0, 1, 2, 3]:
                for scale in [1, 2, 3]:
                    dist_pred = Normal(loc, scale)
                    pit_values = pit(dist_pred, dist_true.sample(n_samples))
                    samples.append(pit_values)
                    annotations.append((dist_pred, dist_true))
    return samples, annotations


def bin_data(pit_values, n_bins):
    return [torch.histc(p, bins=n_bins, min=0, max=1) / len(p) for p in pit_values]


class PITValuesDataset(Dataset):
    def __init__(self, pit_values, annotations):
        self.X, self.y = torch.stack(pit_values), annotations

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return i, self.X[i]


class PITHistDataset(Dataset):
    def __init__(self, pit_values, annotations, n_bins):
        self.X, self.y = torch.stack(bin_data(pit_values, n_bins)), annotations

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.X[i]


@torch.no_grad()
def test(model, data_set):
    data_loader = DataLoader(data_set, batch_size=BS_TEST)
    X_pred = torch.concat([model(x.to(DEVICE)).cpu() for x, _ in data_loader])
    return model.loss(X_pred, data_set.X).item()


@torch.no_grad()
def accuracy(train_embeds, y_train, test_embeds, y_test):
    knn = NearestNeighbors(n_neighbors=1).fit(train_embeds)
    js = knn.kneighbors(test_embeds, return_distance=False)
    n = len(y_test)
    return sum([y_test[i] == y_train[j[0]] for i, j in zip(range(n), js)]) / n


def train(model, train_loader, optimiser):
    for x, y in train_loader:
        optimiser.zero_grad()
        model.loss(model(x.to(DEVICE)), y.to(DEVICE)).backward()
        optimiser.step()


def wasserstein(u_values, v_values):
    u_sorter = torch.argsort(u_values)
    v_sorter = torch.argsort(v_values)
    all_values = torch.cat((u_values, v_values), dim=1)
    all_values, _ = torch.sort(all_values, stable=True)
    # compute the differences between pairs of successive values of u and v
    deltas = torch.diff(all_values)
    # get the respective positions of the values of u and v among the values of both distributions
    batch_idx = torch.arange(u_values.size(0)).reshape(-1, 1)
    u_cdf_indices = torch.searchsorted(u_values[batch_idx, u_sorter], all_values[:, :-1], side='right')
    v_cdf_indices = torch.searchsorted(v_values[batch_idx, v_sorter], all_values[:, :-1], side='right')
    # calculate the CDFs of u and v using their weights, if specified
    u_cdf = u_cdf_indices / u_values.size(1)
    v_cdf = v_cdf_indices / v_values.size(1)
    # compute the value of the integral based on the CDFs
    return torch.sum(torch.multiply(torch.abs(u_cdf - v_cdf), deltas), dim=1, keepdim=True)


class EmbedderDecoder(nn.Module):
    def __init__(self, n_data, embed_dim, n_hiddens, output_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedder = nn.Embedding(n_data, embed_dim).to(DEVICE)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, output_dim),
            nn.Sigmoid()).to(DEVICE)
        self.loss = lambda X_pred, X: wasserstein(X_pred, X).mean()

    def forward(self, i):
        return self.decoder(self.embedder(i))

    @torch.no_grad()
    def embed(self, i):
        return self.embedder(i.to(DEVICE)).cpu()

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x.to(DEVICE)).cpu()

    def new_data_set(self, data_set, hyperparams):
        model = deepcopy(self)
        model.embedder = nn.Embedding(len(data_set), model.embed_dim).to(DEVICE)
        for p in model.decoder.parameters():
            p.requires_grad = False
        model.train_early_stopping(data_set, hyperparams)
        return model

    def train_early_stopping(self, train_set, hyperparams):
        optimiser = Adam(self.parameters(), lr=hyperparams["lr"])
        train_loader = DataLoader(train_set, batch_size=hyperparams["bs"], shuffle=True)
        loss_best = float("inf")
        i = 0
        while i < hyperparams["patience"]:
            train(self, train_loader, optimiser)
            X_train_pred = self.decode(self.embed(torch.arange(len(train_set))))
            loss_train = self.loss(X_train_pred, train_set.X).item()
            if loss_train < loss_best:
                i = 0
                loss_best = loss_train
                model_state_dict_at_best = deepcopy(self.state_dict())
            else:
                i += 1
        self.load_state_dict(model_state_dict_at_best)

    def train_epochs(self, train_set, test_set, hyperparams):
        optimiser = Adam(self.parameters(), lr=hyperparams["lr"])
        train_loader = DataLoader(train_set, batch_size=hyperparams["bs"], shuffle=True)
        for epoch in range(1, hyperparams["epochs"] + 1):
            train(self, train_loader, optimiser)
            train_embeds = self.embed(torch.arange(len(train_set)))
            X_train_pred = self.decode(train_embeds)
            test_model = self.new_data_set(test_set, hyperparams)
            test_embeds = test_model.embed(torch.arange(len(test_set)))
            X_test_pred = test_model.decode(test_embeds)
            wandb.log({
                "train": {"wasserstein": self.loss(X_train_pred, train_set.X).item()},
                "test": {
                    "wasserstein": self.loss(X_test_pred, test_set.X).item(),
                    "accuracy": accuracy(train_embeds, train_set.y, test_embeds, test_set.y)}})


def bhattacharyya(p_hist, q_hist):
    # TODO bug in pytorch? zeros in q_hist produces nans in gradients
    return -(p_hist * (q_hist + 1e-6)).sqrt().sum(1, keepdim=True).log()


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, n_hiddens, embed_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, embed_dim)).to(DEVICE)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, n_hiddens),
            nn.Tanh(),
            nn.Linear(n_hiddens, input_dim),
            nn.Softmax(dim=1)).to(DEVICE)
        self.loss = lambda X_pred, X: bhattacharyya(X_pred, X).mean()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def embed(self, x):
        return self.encoder(x.to(DEVICE)).cpu()

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x.to(DEVICE)).cpu()

    def train_epochs(self, train_set, test_set, hyperparams):
        optimiser = Adam(self.parameters(), lr=hyperparams["lr"])
        train_loader = DataLoader(train_set, batch_size=hyperparams["bs"], shuffle=True)
        for epoch in range(1, hyperparams["epochs"] + 1):
            train(self, train_loader, optimiser)
            train_embeds, test_embeds = self.embed(train_set.X), self.embed(test_set.X)
            X_train_pred, X_test_pred = self.decode(train_embeds), self.decode(test_embeds)
            wandb.log({
                "train": {"bhattacharyya": self.loss(X_train_pred, train_set.X).item()},
                "test": {
                    "bhattacharyya": self.loss(X_test_pred, test_set.X).item(),
                    "accuracy": accuracy(train_embeds, train_set.y, test_embeds, test_set.y)}})


def seed():
    np.random.seed(18)
    torch.manual_seed(16)
    torch.backends.cudnn.benchmark = False


@click.command()
@click.option("--bins", default=0)
@click.option("--bs", default=32)
@click.option("--embed_dim", default=2)
@click.option("--epochs", default=100)
@click.option("--hiddens", default=10)
@click.option("--lr", default=1e-1)
@click.option("--output_dim", default=100)
@click.option("--patience", default=10)
@click.option("--repeats", default=10)
@click.option("--samples", default=1000)
def experiment(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        # reproducibility
        seed()
        # data
        train_data = generate_data(hyperparams["repeats"], hyperparams["samples"])
        test_data = generate_data(n_repeats=1, n_samples=hyperparams["samples"])
        if config["bins"] > 0:
            train_set = PITHistDataset(*train_data, config["bins"])
            test_set = PITHistDataset(*test_data, config["bins"])
            model = EncoderDecoder(config["bins"], config["hiddens"], config["embed_dim"])
        else:
            train_set = PITValuesDataset(*train_data)
            test_set = PITValuesDataset(*test_data)
            model = EmbedderDecoder(len(train_set), config["embed_dim"], config["hiddens"], config["output_dim"])
        # train
        model.train_epochs(train_set, test_set, config)
        torch.save(model.state_dict(), MODELPATH.format(run.name))


if __name__ == "__main__":
    experiment()
