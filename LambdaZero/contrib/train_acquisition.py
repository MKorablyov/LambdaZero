import numpy as np
from ray import tune

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def evaluate_regression(regressor,
                        X,
                        y,
                        samples=100,
                        std_multiplier=2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()


# class EnsembleMLP:
    # make ensemble of weights in torch
    # def fit(x, y, x_val, y_val)
        # this would fit to the dataset

    # def get_mean_and_variance()
        # samples = ensemble(w) @ X
        # return  mean, variance(samples)

class UCB:
    # todo: acquisition function - maybe create classes
    #  AcqusitionFunction; ModelWithUncertainty
    def __init__(self):
        self.model = BayesianRegressor(1024, 1)
        self.seen_x, self.seen_y = None, None
        self.val_x, self.val_y = None, None

    def update_with_seen(self, x, y):
        # self.seen_x +=x
        # self.model_with_uncertainty.fit(x,y, self.val_x, self.val_y)
        pass

    def acqusition_values(self, x):
        # mean, var = self.model.get_mean_and_variance(molecules)
        # return mean + kappa * var
        pass

    def acquire_batch(self, x, discounts, aq_values=None):
        # if aq_values = self.compute_acquisition_values(x)
        # aq_values_ = aq_values[top_k]
        # return idx
        pass


class UCBTrainable(UCB, tune.Trainable):
    # load dataset of fingerprints

    # self.dataset = load_dataset()
    # _train()

        # return {"acc":0.1}
    pass



def load_dataset():
    x = np.random.uniform(size=(10000,1024))
    func = np.random.uniform(size=(1024,1)) / 1024.
    y = np.matmul(x, func)
    return x,y


def standardize_dataset(x, y):
    x = StandardScaler().fit_transform(x)
    # y = StandardScaler().fit_transform(np.expand_dims(y, -1))
    y = StandardScaler().fit_transform(y)
    return x, y


def split_dataset(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
    x_train, y_train = torch.tensor(x_train).float(), torch.tensor(y_train).float()
    x_test, y_test = torch.tensor(x_test).float(), torch.tensor(y_test).float()
    return x_train, y_train, x_test, y_test


x, y = load_dataset()
print("loaded dataset")
x, y = standardize_dataset(x, y)
print("standardized dataset")
x_train, y_train, x_test, y_test = split_dataset(x, y)
print("splitted dataset")


@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.blinear1 = BayesianLinear(input_dim, 512)
        self.blinear2 = BayesianLinear(512, output_dim)

    def forward(self, x):
        x = self.blinear1(x)
        x = self.blinear2(x)
        return x


regressor = BayesianRegressor(1024, 1)

# class UCBTrainer(UCB, tune.trainable):
    # _init():
    #   seen_x, seen_y, val_x, val_y, unseen_x, unseen_y = .....
    # def train()
    #   idx  = self.acquire_batch(unseen_x)
    #   x_, y_ = unseen[idx], unseen[idx]
    #   self.update_with_seen(x_, y_)

# tune run reference here:
# https://github.com/MKorablyov/LambdaZero/blob/master/LambdaZero/examples/bayesian_models/bayes_tune/UCB.py

