import pandas
from sklearn import model_selection
import torch

from . import dist
from . import method


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
    # Year Prediction MSD
    # https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
    df = pandas.read_csv("data/YearPredictionMSD.txt", header=None)
    return df.values[:, 1:], df.values[:, :1]


def array2tensor(array):
    return torch.from_numpy(array.astype("float32"))


class StandardScaler:
    def __init__(self):
        pass

    def fit(self, X):
        self.mean = torch.mean(X, dim=0)
        self.sd = torch.std(X, dim=0)
        self.variance = self.sd**2
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
        self.y_original = y.clone()
        self.y = self.y_scaler.transform(y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)

    def evaluate(self, model):
        alpha, mu, sigma = method.predict(model, self.X)
        mu = self.y_scaler.inverse_transform(mu)
        sigma = self.y_scaler.inverse_transform_sigma(sigma)
        log = {
            "crps": dist.crps_gaussian_mixture(
                self.y_original, alpha, mu, sigma
            ).mean(),
            "nll": dist.nll_gaussian_mixture(self.y_original, alpha, mu, sigma).mean(),
        }
        log["loss"] = log["nll"]
        return log


def split(X, y, seed, device=None):
    split_test = model_selection.train_test_split(
        X, y, test_size=0.1, random_state=seed
    )
    X_train, X_test, y_train, y_test = split_test
    split_valid = model_selection.train_test_split(
        X_train, y_train, test_size=0.1, random_state=79
    )
    X_train, X_valid, y_train, y_valid = split_valid
    trainset = UCIDataset(X_train, y_train, device=device)
    validset = UCIDataset(
        X_valid, y_valid, trainset.X_scaler, trainset.y_scaler, device
    )
    testset = UCIDataset(X_test, y_test, trainset.X_scaler, trainset.y_scaler, device)
    return trainset, validset, testset
