import torch


def seed():
    torch.manual_seed(16)
    torch.backends.cudnn.benchmark = False
