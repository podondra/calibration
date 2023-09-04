import click
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import wandb

from calibration import data
from calibration import method
from calibration import pit


@click.group()
def cli():
    pass


@cli.command()
@click.option("--bins", default=20)
@click.option("--bs", default=64)
@click.option("--device", default="cuda")
@click.option("--m", default=5)
@click.option("--lr", default=1e-2)
@click.option("--neurons", default=16)
@click.option("--patience", default=1000)
@click.option("--seed", default=16)
def interpreter(**hyperparams):
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        device = torch.device(config["device"])
        valids = 1000
        validset = pit.PITDataset(valids, config["bins"], device=device)
        trainset = pit.PITSampler(config["bs"], config["bins"], device=device)
        loader = torch.utils.data.DataLoader(trainset, config["bs"])
        model = method.MDN(config["bins"], config["neurons"], config["m"])
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
        method.early_stopping(model, loader, trainset, validset, optimiser, config)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparams": dict(config)}, f"models/{run.name}.pt")


@cli.command()
@click.option("--bs", default=100)
@click.option("--device", default="cuda")
@click.option("--lr", default=1e-3)
@click.option("--m", default=1)
@click.option("--neurons", default=100)
@click.option("--patience", default=100)
@click.option("--seed", default=16)
def regressor(**hyperparams):
    X_train, y_train = data.protein()
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                        test_size=0.1,
                                                        random_state=33)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                          test_size=0.1,
                                                          random_state=79)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)
    X_valid = X_scaler.transform(X_valid)
    y_valid = y_scaler.transform(y_valid)
    hyperparams["inputs"] = X_train.shape[-1]
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        device = torch.device(config["device"])
        trainset = data.Dataset(X_train, y_train, device)
        validset = data.Dataset(X_valid, y_valid, device)
        loader = torch.utils.data.DataLoader(trainset, config["bs"], shuffle=True)
        #model = method.MDN(config["inputs"], config["neurons"], config["m"])
        model = method.DE(config["inputs"], config["neurons"], config["m"])
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
        method.early_stopping(model, loader, trainset, validset, optimiser, config)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparams": dict(config)}, f"models/{run.name}.pt")


if __name__ == "__main__":
    cli()
