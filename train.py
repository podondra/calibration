import click
import torch
import wandb

from calibration import data
from calibration import dist
from calibration import method
from calibration import pit


DATANAME = {"power": data.power, "protein": data.protein, "year": data.year}


@click.group()
@click.option("--bs", default=100)
@click.option("--device", default="cuda")
@click.option("--lr", default=1e-3)
@click.option("--patience", default=100)
@click.option("--seed", default=4)
@click.pass_context
def train(context, **hyperparams):
    context.ensure_object(dict)
    context.obj = hyperparams
    torch.manual_seed(hyperparams["seed"])
    torch.backends.cudnn.benchmark = False


@train.command()
@click.option("--bins", default=20)
@click.option("--components", default=5)
@click.option("--neurons", default=16)
@click.pass_context
def interpreter(context, **hyperparams):
    hyperparams |= context.obj
    with wandb.init(config=hyperparams) as run:
        device = torch.device(hyperparams["device"])
        valids = 1000
        validset = pit.PITDataset(valids, hyperparams["bins"], device=device)
        trainset = pit.PITSampler(hyperparams["bs"], hyperparams["bins"], device=device)
        loader = torch.utils.data.DataLoader(trainset, hyperparams["bs"])
        model = method.MDN(
            hyperparams["bins"],
            hyperparams["neurons"],
            hyperparams["components"],
            loss=method.wasserstein_loss,
        )
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        method.early_stopping(model, loader, trainset, validset, optimiser, hyperparams)
        torch.save(
            {"model_state_dict": model.state_dict(), "hyperparams": dict(hyperparams)},
            f"models/{run.name}.pt",
        )


def experiment(Model, hyperparams_model, hyperparams):
    hyperparams["dataname"] = hyperparams_model["dataname"]
    del hyperparams_model["dataname"]
    X, y = DATANAME[hyperparams["dataname"]]()
    hyperparams_model["inputs"] = X.shape[-1]
    hyperparams |= hyperparams_model
    with wandb.init(config=hyperparams) as run:
        device = torch.device(hyperparams["device"])
        trainset, validset, testset = data.split(X, y, hyperparams["seed"], device)
        loader = torch.utils.data.DataLoader(trainset, hyperparams["bs"], shuffle=True)
        model = Model(**hyperparams_model)
        model = model.to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        method.early_stopping(model, loader, trainset, validset, optimiser, hyperparams)
        log_test = testset.evaluate(model)
        wandb.run.summary["test.nll"] = log_test["nll"]
        wandb.run.summary["test.crps"] = log_test["crps"]
        torch.save(
            {"model_state_dict": model.state_dict(), "hyperparams": dict(hyperparams)},
            f"models/{wandb.run.name}.pt",
        )


@train.command()
@click.option("--neurons", default=100)
@click.argument("dataname")
@click.pass_context
def dn(context, **hyperparams):
    context.obj["method"] = "dn"
    hyperparams["components"] = 1
    experiment(method.MDN, hyperparams, context.obj)


@train.command()
@click.option("--components", default=5)
@click.option("--neurons", default=100)
@click.argument("dataname")
@click.pass_context
def mdn(context, **hyperparams):
    context.obj["method"] = "mdn"
    experiment(method.MDN, hyperparams, context.obj)


@train.command()
@click.option("--members", default=5)
@click.option("--neurons", default=100)
@click.argument("dataname")
@click.pass_context
def de(context, **hyperparams):
    context.obj["method"] = "de"
    experiment(method.DE, hyperparams, context.obj)


if __name__ == "__main__":
    train(obj={})
