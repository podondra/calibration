import pandas
import torch


def array2tensor(array):
    return torch.from_numpy(array.astype("float32"))


def protein():
    # Physicochemical Properties of Protein Tertiary Structure
    # https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
    df = pandas.read_csv("data/CASP.csv")
    return df.values[:, 1:], df.values[:, :1]


def power():
    # Combined Cycle Power Plant
    # https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
    df = pd.read_excel("data/Folds5x2_pp.xlsx")
    return df.values[:, :4], df.values[:, -1:]


def year():
    # YearPredictionMSD
    # https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
    #
    # TODO
    # You should respect the following train / test split:
    # train: first 463,715 examples
    # test: last 51,630 examples
    # It avoids the 'producer effect' by making sure no song
    # from a given artist ends up in both the train and test set.
    df = pd.read_csv("data/YearPredictionMSD.txt", header=None)
    return df.values[:, 1:], df.values[:, :1]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, device=None):
        self.X = array2tensor(X).to(device)
        self.y = array2tensor(y).to(device)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)
