import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import wandb


def train_epochs(model, trainset, testset, hyperparams):
    optimiser = Adam(model.parameters(), lr=hyperparams["lr"])
    scheduler = StepLR(optimiser, step_size=hyperparams["step"],
                       gamma=hyperparams["gamma"])
    loader = DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
    for epoch in range(1, hyperparams["epochs"] + 1):
        log_train = model.train(loader, optimiser)
        log_test = model.test(testset)
        wandb.log({"train": log_train, "test": log_test})
        scheduler.step()


def kl_divergence(mu, sigma):
    return 0.5 * (sigma.square() + mu.square() - 2 * sigma.log() - 1).sum(1, keepdim=True)


class VAE(nn.Module):
    def __init__(self, input_dim, n_hiddens, n_neurons, embed_dim, epsilon):
        super().__init__()
        # hyperparams
        self.embed_dim = embed_dim
        self.epsilon = epsilon
        self.loss_rec = nn.MSELoss(reduction="none")
        # encoder
        encoder = [nn.Linear(input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hiddens - 1):
            encoder += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        self.encoder = nn.Sequential(*encoder)
        self.fc_mu = nn.Linear(n_neurons, embed_dim)
        # TODO output log_var?
        self.fc_sigma = nn.Sequential(nn.Linear(n_neurons, embed_dim), nn.Softplus())
        # decoder
        decoder = [nn.Linear(embed_dim, n_neurons), nn.Tanh()]
        for _ in range(n_hiddens - 1):
            decoder += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        decoder += [nn.Linear(n_neurons, input_dim)]
        # TODO decoder += [nn.Softmax(dim=1)]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        x = self.encoder(x)
        mu, sigma = self.fc_mu(x), self.fc_sigma(x)
        z = mu + sigma * torch.randn_like(mu)
        return self.decoder(z), mu, sigma

    @torch.no_grad()
    def encode(self, x):
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_sigma(x)

    @torch.no_grad()
    def decode(self, x):
        return self.decoder(x)

    def train(self, loader, optimiser):
        # TODO correct running loss log
        for x in loader:
            optimiser.zero_grad()
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
        X_pred, mu, sigma = self(dataset.X)
        loss_rec = self.loss_rec(X_pred, dataset.X).mean()
        kl_div = kl_divergence(mu, sigma).mean()
        loss = loss_rec + self.epsilon * kl_div
        return {"reconstruction": loss_rec.item(),
                "kl_divergence": kl_div.item(),
                "loss": loss.item()}
