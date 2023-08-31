import copy

import torch
import wandb


def train_early_stopping(model, loader, validset, optimiser, hyperparams):
    loss_best = float("inf")
    i = 0
    while i < hyperparams["patience"]:
        log_train = model.train(loader, optimiser)
        log_valid = model.evaluate(validset)
        if log_valid["loss"] < loss_best:
            loss_best = log_valid["loss"]
            model_state_dict_best = copy.deepcopy(model.state_dict())
            i = 0
        else:
            i += 1
        wandb.log({"train": log_train, "valid": log_valid})
        wandb.run.summary["valid.loss"] = loss_best
    model.load_state_dict(model_state_dict_best)
