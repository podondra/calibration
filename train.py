import click
from torchvision import datasets
import torch
from torch import optim
from torch.optim import lr_scheduler
import wandb

from calibration import data
from calibration import utils
from calibration import vae


@click.command()
@click.option("--beta", default=1.0)
@click.option("--bins", default=20)
@click.option("--bs", default=100)
@click.option("--device", default="cuda")
@click.option("--gamma", default=1.0)
@click.option("--latents", default=4)
@click.option("--lr", default=1e-3)
@click.option("--neurons", default=16)
@click.option("--patience", default=5000)
@click.option("--seed", default=16)
@click.option("--step", default=1000)
@click.option("--wd", default=0.0)    # TODO Kingma & Welling (2014) used weigth decay
def train(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        device = torch.device(config["device"])
        utils.seed(config["seed"])
        validset = data.PITHistDataset(config["bins"], device=device)
        trainset = data.PITHistSampler(config["bs"], config["bins"], device=device)
        model = vae.VAE(config["bins"], config["neurons"], config["latents"], config["beta"])
        model = model.to(device)
        optimiser = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = lr_scheduler.StepLR(optimiser, step_size=config["step"], gamma=config["gamma"])
        vae.train_early_stopping(model, trainset, validset, optimiser, scheduler, config)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparams": hyperparams}, f"models/{run.name}.pt")


if __name__ == "__main__":
    train()
