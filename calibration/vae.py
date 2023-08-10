import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb

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
    def __init__(self, input_dim, n_hiddens, n_neurons, embed_dim, epsilon, device):
        super().__init__()
        # hyperparams
        self.embed_dim = embed_dim
        self.epsilon = epsilon
        self.loss_rec = mse    # TODO
        self.device = device
        # encoder
        encoder = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hiddens - 1):
            encoder += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        #encoder += [nn.Linear(n_neurons, embed_dim + 1)]
        self.encoder = nn.Sequential(*encoder).to(device)
        self.fc_mu = nn.Linear(n_neurons, embed_dim).to(device)
        self.fc_sigma = nn.Sequential(nn.Linear(n_neurons, embed_dim), nn.Softplus()).to(device)
        # decoder
        decoder = [nn.Linear(embed_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hiddens - 1):
            decoder += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        decoder += [nn.Linear(n_neurons, input_dim)]
        #decoder += [nn.Softmax(dim=1)]    # TODO
        self.decoder = nn.Sequential(*decoder).to(device)

    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.fc_mu(x), self.fc_sigma(x)
        z = mu + sigma * torch.randn_like(mu)
        return self.decoder(z), mu, sigma

    @torch.no_grad()
    def encode(self, x):
        x = self.encoder(x.to(self.device))
        return self.fc_mu(x).cpu(), self.fc_sigma(x).cpu()

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x.to(self.device)).cpu()# TODO .softmax(dim=1)

    def train(self, loader, optimiser):
        for x in loader:
            optimiser.zero_grad()
            x = x.to(self.device)
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
        X_pred, mu, sigma = self(dataset.X.to(self.device))
        X_pred, mu, sigma = X_pred.cpu(), mu.cpu(), sigma.cpu()
        loss_rec = self.loss_rec(X_pred, dataset.X).mean()
        kl_div = kl_divergence(mu, sigma).mean()
        loss = loss_rec + self.epsilon * kl_div
        return {"reconstruction": loss_rec.item(),
                "kl_divergence": kl_div.item(),
                "loss": loss.item()}
