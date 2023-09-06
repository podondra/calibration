import click
from sklearn.model_selection import train_test_split
import torch
import wandb

from calibration import data
from calibration import method
from calibration import pit


DATANAME = {"power": data.power,
            "protein": data.protein,
            "year": data.year}


@click.group()
@click.option("--bs", default=100)
@click.option("--device", default="cuda")
@click.option("--lr", default=1e-3)
@click.option("--patience", default=100)
@click.pass_context
def train(ctx, **hyperparams):
    ctx.ensure_object(dict)
    ctx.obj = hyperparams


@train.command()
@click.option("--bins", default=20)
@click.option("--components", default=5)
@click.option("--neurons", default=16)
@click.option("--seed", default=16)
@click.pass_context
def interpreter(ctx, **hyperparams):
    hyperparams |= ctx.obj
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.benchmark = False
        device = torch.device(config["device"])
        valids = 1000
        validset = pit.PITDataset(valids, config["bins"], device=device)
        trainset = pit.PITSampler(config["bs"], config["bins"], device=device)
        loader = torch.utils.data.DataLoader(trainset, config["bs"])
        model = method.MDN(config["bins"], config["neurons"], config["components"])
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=config["lr"])
        method.early_stopping(model, loader, trainset, validset, optimiser, config)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparams": dict(config)}, f"models/{run.name}.pt")


def experiment(model, X, y, hyperparams):
    split_test = train_test_split(X, y, test_size=0.1, random_state=hyperparams["seed"])
    X_train, X_test, y_train, y_test = split_test
    split_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=79)
    X_train, X_valid, y_train, y_valid = split_valid
    torch.manual_seed(hyperparams["seed"])
    torch.backends.cudnn.benchmark = False
    device = torch.device(hyperparams["device"])
    trainset = data.UCIDataset(X_train, y_train, device=device)
    validset = data.UCIDataset(X_valid, y_valid, trainset.X_scaler, trainset.y_scaler, device)
    loader = torch.utils.data.DataLoader(trainset, hyperparams["bs"], shuffle=True)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    method.early_stopping(model, loader, trainset, validset, optimiser, hyperparams)
    testset = data.UCIDataset(X_test, y_test, trainset.X_scaler, trainset.y_scaler, device)
    log_test = testset.evaluate(model)
    wandb.run.summary["test.nll"] = log_test["nll"]
    wandb.run.summary["test.crps"] = log_test["crps"]
    torch.save({"model_state_dict": model.state_dict(),
                "hyperparams": dict(hyperparams)}, f"models/{wandb.run.name}.pt")


@train.command()
@click.option("--neurons", default=100)
@click.option("--seed", default=16)
@click.argument("dataname")
@click.pass_context
def dn(ctx, **hyperparams):
    hyperparams["method"] = "dn"
    hyperparams |= ctx.obj
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        X, y = DATANAME[hyperparams["dataname"]]()
        model = method.MDN(X.shape[-1], config["neurons"], 1)
        experiment(model, X, y, config)


@train.command()
@click.option("--components", default=5)
@click.option("--neurons", default=100)
@click.option("--seed", default=16)
@click.argument("dataname")
@click.pass_context
def mdn(ctx, **hyperparams):
    hyperparams["method"] = "mdn"
    hyperparams |= ctx.obj
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        X, y = DATANAME[hyperparams["dataname"]]()
        model = method.MDN(X.shape[-1], config["neurons"], config["components"])
        experiment(model, X, y, config)


@train.command()
@click.option("--members", default=5)
@click.option("--neurons", default=100)
@click.option("--seed", default=16)
@click.argument("dataname")
@click.pass_context
def de(ctx, **hyperparams):
    hyperparams["method"] = "de"
    hyperparams |= ctx.obj
    with wandb.init(config=hyperparams) as run:
        config = wandb.config
        X, y = DATANAME[hyperparams["dataname"]]()
        model = method.DE(X.shape[-1], config["neurons"], config["members"])
        experiment(model, X, y, config)


if __name__ == "__main__":
    train(obj={})
