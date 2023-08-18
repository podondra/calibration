import click
from torchvision import datasets
import torch
import wandb

from calibration import data
from calibration import utils
from calibration import vae


@click.command()
@click.option("--bins", default=20)
@click.option("--bs", default=64)
@click.option("--device", default="cuda")
@click.option("--embeds", default=2)
@click.option("--epochs", default=1000)
@click.option("--epsilon", default=1e-5)
@click.option("--gamma", default=1.0)
@click.option("--hiddens", default=1)
@click.option("--log", default=10)
@click.option("--lr", default=1e-2)
@click.option("--neurons", default=16)
@click.option("--patience", default=500)
@click.option("--samples", default=10000)
@click.option("--seed", default=16)
@click.option("--step", default=1000)
@click.option("--wd", default=0.0)
def train(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        device = torch.device(config["device"])
        utils.seed(config["seed"])
        testset = data.PITHistDataset(config["samples"], config["bins"],
                                      device)
        trainset = data.PITHistSampler(config["log"] * config["bs"],
                                       config["samples"], config["bins"],
                                       device)
        model = vae.VAE(config["bins"], config["hiddens"], config["neurons"],
                        config["embeds"], config["epsilon"])
        model = model.to(device)
        vae.train_early_stopping(model, trainset, testset, config)
        torch.save(model.state_dict(), f"models/{run.name}.pt")


if __name__ == "__main__":
    train()
