import pandas
import torch


def protein():
    # Physicochemical Properties of Protein Tertiary Structure
    # https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
    df = pandas.read_csv("data/CASP.csv")
    return df.values[:, 1:], df.values[:, :1]


def power():
    # Combined Cycle Power Plant
    # https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
    df = pandas.read_excel("data/Folds5x2_pp.xlsx")
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
    df = pandas.read_csv("data/YearPredictionMSD.txt", header=None)
    return df.values[:, 1:], df.values[:, :1]


def array2tensor(array):
    return torch.from_numpy(array.astype("float32"))


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = torch.mean(X, dim=0)
        # TODO verify unbiased
        self.sd = torch.std(X, dim=0, unbiased=True)
        # TODO self.sd[self.sd == 0.0] = 1.0
        self.variance = self.sd ** 2
        return self

    def transform(self, X):
        return (X - self.mean) / self.sd

    def inverse_transform(self, X):
        return self.sd * X + self.mean

    def inverse_transform_sigma(self, sigma):
        return self.variance * sigma


class UCIDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, X_scaler=None, y_scaler=None, device=None):
        X = array2tensor(X).to(device)
        y = array2tensor(y).to(device)
        self.X_scaler, self.y_scaler = X_scaler, y_scaler
        if X_scaler is None:
            self.X_scaler = StandardScaler().fit(X)
        if y_scaler is None:
            self.y_scaler = StandardScaler().fit(y)
        self.X = self.X_scaler.transform(X)
        self.y = self.y_scaler.transform(y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)
