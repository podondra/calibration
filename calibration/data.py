import pandas
from sklearn import model_selection
import torch

from . import dist
from . import method


def array2tensor(array):
    return torch.from_numpy(array.astype("float32"))


def protein():
    # Physicochemical Properties of Protein Tertiary Structure
    # https://archive.ics.uci.edu/dataset/265/physicochemical+properties+of+protein+tertiary+structure
    df = pandas.read_csv("data/CASP.csv")
    tensor = array2tensor(df.values)
    return tensor[:, 1:], tensor[:, :1]


def power():
    # Combined Cycle Power Plant
    # https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
    df = pandas.read_excel("data/Folds5x2_pp.xlsx")
    tensor = array2tensor(df.values)
    return tensor[:, :4], tensor[:, -1:]


def year():
    # Year Prediction MSD
    # https://archive.ics.uci.edu/dataset/203/yearpredictionmsd
    df = pandas.read_csv("data/YearPredictionMSD.txt", header=None)
    tensor = array2tensor(df.values)
    return tensor[:, 1:], tensor[:, :1]


def synthetic():
    # generate data by running $ python generate.py
    df = pandas.read_csv("data/synthetic.csv")
    tensor = array2tensor(df.values)
    return tensor[:, :1], tensor[:, 1:]


class StandardScaler:
    def __init__(self, device=None):
        self.device = device

    def fit(self, X):
        self.mean = torch.mean(X, dim=0).to(self.device)
        self.sd = torch.std(X, dim=0).to(self.device)
        self.variance = self.sd**2
        return self

    def transform(self, X):
        return (X - self.mean) / self.sd

    def inverse_transform(self, X):
        return self.sd * X + self.mean

    def inverse_transform_sigma(self, sigma):
        return self.variance * sigma


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y, X_scaler, y_scaler, device=None):
        X, y = X.to(device), y.to(device)
        if X_scaler is not None:
            X = X_scaler.transform(X)
        self.y_original = y.clone()
        if y_scaler is not None:
            y = y_scaler.transform(y)
        self.X, self.y = X, y
        self.X_scaler, self.y_scaler = X_scaler, y_scaler

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)

    def evaluate(self, model):
        alpha, mu, sigma = method.predict(model, self.X)
        if self.y_scaler is not None:
            mu = self.y_scaler.inverse_transform(mu)
            sigma = self.y_scaler.inverse_transform_sigma(sigma)
        log = {
            "crps": dist.crps_gaussian_mixture(self.y_original, alpha, mu, sigma).mean(),
            "nll": dist.nll_gaussian_mixture(self.y_original, alpha, mu, sigma).mean(),
        }
        log["loss"] = log["nll"]
        return log


def split(X, y, seed, scale, device=None):
    split_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=seed)
    X_train, X_test, y_train, y_test = split_test
    split_valid = model_selection.train_test_split(X_train, y_train, test_size=0.1, random_state=79)
    X_train, X_valid, y_train, y_valid = split_valid
    X_scaler = StandardScaler(device).fit(X_train) if scale else None
    y_scaler = StandardScaler(device).fit(y_train) if scale else None
    trainset = Dataset(X_train, y_train, X_scaler, y_scaler, device)
    validset = Dataset(X_valid, y_valid, X_scaler, y_scaler, device)
    testset = Dataset(X_test, y_test, X_scaler, y_scaler, device)
    return trainset, validset, testset
