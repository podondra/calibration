import click
import torch
import wandb

from calibration import data
from calibration import utils
from calibration import vae


@click.command()
# data hyperparams
@click.option("--bins", default=10)
@click.option("--samples", default=1000)
# training hyperparams
@click.option("--bs", default=32)
@click.option("--device", default="cuda")
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
def train(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        device = torch.device(config["device"])
        trainset = data.PITHistSampler(config["bs"], config["samples"], config["bins"])
        utils.seed()    # reproducibility
        testset = data.PITHistDataset(config["samples"], config["bins"])
        model = vae.VAE(config["bins"],
                config["hiddens"],
                config["neurons"],
                config["embed"],
                config["epsilon"],
                device)
        if config["modelfile"] is not None:
            model.load_state_dict(torch.load(config["modelfile"], map_location=device))    # TODO load optimiser state
        vae.train_epochs(model, trainset, testset, config)
        torch.save(model.state_dict(), "models/{}.pt".format(run.name))


if __name__ == "__main__":
    train()
