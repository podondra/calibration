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
@click.option("--bins", default=20)
@click.option("--bs", default=64)
@click.option("--device", default="cuda")
@click.option("--embeds", default=3)
@click.option("--epsilon", default=1e-5)
@click.option("--gamma", default=1.0)
@click.option("--hiddens", default=1)
@click.option("--lr", default=1e-2)
@click.option("--neurons", default=16)
@click.option("--patience", default=5000)
@click.option("--samples", default=10000)
@click.option("--seed", default=16)
@click.option("--step", default=1000)
@click.option("--wd", default=0.0)
def train(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        device = torch.device(config["device"])
        utils.seed(config["seed"])
        testset = data.PITHistDataset(config["samples"], config["bins"], device)
        trainset = data.PITHistSampler(config["bs"], config["samples"], config["bins"], device)
        model = vae.VAE(config["bins"], config["hiddens"], config["neurons"],
                        config["embeds"], config["epsilon"])
        model = model.to(device)
        optimiser = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = lr_scheduler.StepLR(optimiser, step_size=config["step"], gamma=config["gamma"])
        vae.train_early_stopping(model, trainset, testset, optimiser, scheduler, config)
        torch.save(model.state_dict(), f"models/{run.name}.pt")


if __name__ == "__main__":
    train()
