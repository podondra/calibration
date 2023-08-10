import numpy
import torch


def seed():
    numpy.random.seed(18)
    torch.manual_seed(16)
    torch.backends.cudnn.benchmark = False


