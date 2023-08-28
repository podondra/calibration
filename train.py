import click
from torchvision import datasets
import torch
from torch import optim
from torch.optim import lr_scheduler
import wandb

from calibration import data
from calibration import mdn
from calibration import train


@click.command()
@click.option("--bins", default=20)
@click.option("--bs", default=64)
@click.option("--device", default="cuda")
@click.option("--gamma", default=1.0)
@click.option("--hiddens", default=1)
@click.option("--k", default=5)
@click.option("--lr", default=1e-3)
@click.option("--neurons", default=16)
@click.option("--patience", default=1000)
@click.option("--seed", default=16)
@click.option("--step", default=1000)
@click.option("--wd", default=0.0)
def main(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        device = torch.device(config["device"])
        validset = data.PITHistRndDataset(n=1000, bins=config["bins"], device=device)
        trainset = data.PITHistSampler(bs=config["bs"], bins=config["bins"], device=device)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["bs"])
        model = mdn.MDN(config["bins"], config["neurons"], config["hiddens"], config["k"])
        model = model.to(device)
        optimiser = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
        scheduler = lr_scheduler.StepLR(optimiser, step_size=config["step"], gamma=config["gamma"])
        train.train_early_stopping(model, trainloader, validset, optimiser, scheduler, config)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparams": hyperparams}, f"models/{run.name}.pt")


if __name__ == "__main__":
    main()
