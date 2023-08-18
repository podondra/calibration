import torch


def seed(seed=16):
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
