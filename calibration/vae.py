import copy

import torch
import wandb


def train_early_stopping(model, trainset, validset, optimiser, scheduler, hyperparams):
    loader = torch.utils.data.DataLoader(trainset, batch_size=hyperparams["bs"], shuffle=True)
    elbo_best = float("-inf")
    i = 0
    while i < hyperparams["patience"]:
        log_train = model.train(loader, optimiser)
        log_valid = model.evaluate(validset)
        if log_valid["elbo"] > elbo_best:
            elbo_best = log_valid["elbo"]
            model_state_dict_best = copy.deepcopy(model.state_dict())
            i = 0
        else:
            i += 1
        wandb.log({"train": log_train, "valid": log_valid})
        wandb.run.summary["valid.elbo"] = elbo_best
        scheduler.step()
    model.load_state_dict(model_state_dict_best)


class Encoder(torch.nn.Module):
    def __init__(self, inputs, neurons, latents):
        super().__init__()
        self.fc_hidden = torch.nn.Linear(inputs, neurons)
        self.fc_mu = torch.nn.Linear(neurons, latents)
        self.fc_ln_var = torch.nn.Linear(neurons, latents)

    def forward(self, x):
        h = torch.tanh(self.fc_hidden(x))
        mu = self.fc_mu(h)
        ln_var = self.fc_ln_var(h)
        return mu, ln_var

    @torch.no_grad()
    def encode(self, x):
        mu, ln_var = self(x)
        sigma = torch.exp(0.5 * ln_var)
        return mu, sigma


class Decoder(torch.nn.Module):
    def __init__(self, latents, neurons, outputs):
        super().__init__()
        self.fc_hidden = torch.nn.Linear(latents, neurons)
        self.fc_mu = torch.nn.Linear(neurons, outputs)
        self.fc_ln_var = torch.nn.Linear(neurons, outputs)

    def forward(self, z):
        h = torch.tanh(self.fc_hidden(z))
        mu = torch.softmax(self.fc_mu(h), dim=-1)
        ln_var = self.fc_ln_var(h)
        return mu, ln_var

    @torch.no_grad()
    def decode(self, z):
        mu, ln_var = self(z)
        sigma = torch.exp(0.5 * ln_var)
        return mu, sigma


def kl_divergence(mu, ln_var):
    return -0.5 * torch.sum(1 + ln_var - mu ** 2 - ln_var.exp(), dim=-1)


def likelihood(mu, ln_var, x):
    # L = 1 according to Kingma & Welling (2014)
    covariance_matrix = torch.diag_embed(torch.exp(ln_var))
    m = torch.distributions.MultivariateNormal(mu, covariance_matrix)
    return m.log_prob(x)


class VAE(torch.nn.Module):
    def __init__(self, inputs, neurons, latents, beta=1):
        super().__init__()
        self.beta = beta
        self.encoder = Encoder(inputs, neurons, latents)
        self.decoder = Decoder(latents, neurons, inputs)

    def encode(self, x):
        return self.encoder.encode(x)

    def decode(self, z):
        return self.decoder.decode(z)

    def rsample(self, mu, ln_var):
        return mu + torch.exp(0.5 * ln_var) * torch.randn_like(mu)

    def elbo(self, ln_pxz, kl):
        return torch.mean(ln_pxz - self.beta * kl)

    def train(self, loader, optimiser):
        for x_noisy, x, _ in loader:
            optimiser.zero_grad()
            # forward
            mu_z, ln_var_z = self.encoder(x_noisy)
            z = self.rsample(mu_z, ln_var_z)
            mu_x, ln_var_x = self.decoder(z)
            kl = kl_divergence(mu_z, ln_var_z)
            ln_pxz = likelihood(mu_x, ln_var_x, x)
            elbo = self.elbo(ln_pxz, kl)
            loss = -elbo
            # backward
            loss.backward()
            optimiser.step()
        # the sampler has always only single batch
        return {"elbo": elbo.item(),
                "kl": kl.detach().mean().item(),
                "ln_pxz": ln_pxz.detach().mean().item()}

    @torch.no_grad()
    def evaluate(self, dataset):
        mu_z, ln_var_z = self.encoder(dataset.X_noisy)
        z = self.rsample(mu_z, ln_var_z)
        mu_x, ln_var_x = self.decoder(z)
        kl = kl_divergence(mu_z, ln_var_z)
        ln_pxz = likelihood(mu_x, ln_var_x, dataset.X)
        elbo = self.elbo(ln_pxz, kl)
        return {"elbo": elbo.item(),
                "kl": kl.detach().mean().item(),
                "ln_pxz": ln_pxz.detach().mean().item()}
