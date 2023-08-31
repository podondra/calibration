import click
import torch
import wandb

from calibration import data
from calibration import mdn
from calibration import train


@click.command()
@click.option("--bins", default=20)
@click.option("--bs", default=64)
@click.option("--device", default="cuda")
@click.option("--m", default=5)
@click.option("--lr", default=1e-2)
@click.option("--neurons", default=16)
@click.option("--patience", default=1000)
@click.option("--seed", default=16)
def main(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        device = torch.device(config["device"])
        valids = 1000
        validset = data.PITDataset(valids, config["bins"], device=device)
        trainset = data.PITSampler(config["bs"], config["bins"], device=device)
        loader = torch.utils.data.DataLoader(trainset, config["bs"])
        model = mdn.MDN(config["bins"], config["neurons"], config["m"])
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
        train.train_early_stopping(model, loader, validset, optimiser, config)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparams": hyperparams}, f"models/{run.name}.pt")


if __name__ == "__main__":
    main()
