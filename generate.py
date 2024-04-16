import csv
import math

import click
import torch


@click.command()
@click.option("--n", default=10000)
@click.option("--seed", default=75)
def generate(n, seed):
    y = -1 + 2 * torch.rand(n, 1)
    X = y ** 2
    torch.manual_seed(seed)
    y += 0.25 * torch.randn_like(y)
    with open("data/synthetic.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "y"])
        writer.writeheader()
        writer.writerows({"X": x_.item(), "y": y_.item()} for x_, y_ in zip(X, y))


if __name__ == "__main__":
    generate()
