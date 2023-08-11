import click
import torch
import wandb

from calibration import data
from calibration import utils
from calibration import vae


@click.command()
@click.option("--bins", default=10)
@click.option("--bs", default=32)
@click.option("--device", default="cuda")
@click.option("--embed", default=2)
@click.option("--epochs", default=1000)
@click.option("--epsilon", default=1e-3)
@click.option("--gamma", default=1.0)
@click.option("--hiddens", default=1)
@click.option("--lr", default=1e-3)
@click.option("--modelfile", type=click.Path(exists=True))
@click.option("--neurons", default=8)
@click.option("--samples", default=1000)
@click.option("--step", default=1000)
def train(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        device = torch.device(config["device"])
        trainset = data.PITHistSampler(config["bs"], config["samples"],
                                       config["bins"], config["device"])
        utils.seed()    # reproducibility
        testset = data.PITHistDataset(config["samples"], config["bins"],
                                      config["device"])
        model = vae.VAE(config["bins"], config["hiddens"], config["neurons"],
                        config["embed"], config["epsilon"]).to(device)
        if config["modelfile"] is not None:
            checkpoint = torch.load(config["modelfile"], map_location=device)
            model.load_state_dict(checkpoint)
        vae.train_epochs(model, trainset, testset, config)
        torch.save(model.state_dict(), f"models/{run.name}.pt")


if __name__ == "__main__":
    train()
