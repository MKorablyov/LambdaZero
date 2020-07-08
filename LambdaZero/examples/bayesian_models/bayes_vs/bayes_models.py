
import torch
from torch import nn
from torch import distributions
from sklearn import linear_model

from . import chem_ops



class BayesianRegression(nn.Module):
    def __init__(self, embedding_func, fit_mean=True):
        super().__init__()
        self.feature_dim = embedding_func.fp_dim

        self.beta = 1.
        self.alpha = 1.

        self.K = None
        self.chol_K = None
        self.m = None

        self.fit_mean = fit_mean
        self.mean_ = 0.
        self.embedding_func = embedding_func

    def fit(self, x_train, y_train):
        """
        :param x_train: [N, M]
        :param y_train: [N, 1]
        """
        if self.fit_mean:
            self.mean_ = y_train.mean(dim=0)
            y_train = y_train - self.mean_

        #todo: this is a bit of a hack to tune hyperparameters, ideally we should do this ourselves
        # as we potentially want to do it quickly/backprop so would be useful to have it all in pytorch, anyway for now
        # we will cheat by setting the hyperparamters to those suggested by sklearn.
        clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
        clf.fit(x_train.detach().numpy(), y_train.detach().squeeze().numpy())
        self.alpha = clf.lambda_
        self.beta = clf.alpha_
        print(f"Setting hyperparameters to {self.alpha} and {self.beta}")

        assert len(y_train.shape) == 2
        self.K = self.beta * x_train.T @ x_train + self.alpha * torch.eye(self.feature_dim)   # [M, M]
        self.chol_K = torch.cholesky(self.K)

        projected_y = x_train.T @ y_train
        k_inv_projected_y = torch.cholesky_solve(projected_y, self.chol_K)
        self.m = self.beta * k_inv_projected_y  # [M, 1]

    def fit_on_smiles(self, x_train_smiles, y_train):
        feats = self._convert_smiles_to_features(x_train_smiles)
        return self.fit(feats, y_train)

    def predict(self, x_test):
        """
        :param x_test: [N_test, M]
        """
        n_test = x_test.shape[0]
        mean = x_test @ self.m + self.mean_ # [N_test, 1]

        chol_inv_x_tran, _ = torch.triangular_solve(x_test.T, self.chol_K, upper=False)
        variance = chol_inv_x_tran.T @ chol_inv_x_tran + 1./self.beta  # [N_test, N_test]
        variance += 1e-3 * torch.eye(n_test)  # <- this is for numerical stability.
        out = distributions.MultivariateNormal(mean.squeeze(1), variance)
        #todo: this should be changed to predict mean and covariance.

        return out

    def predict_on_smiles(self, x_test_smiles):
        feats = self._convert_smiles_to_features(x_test_smiles)
        return self.predict(feats)

    def _convert_smiles_to_features(self, list_of_smiles):
        return self.embedding_func(list_of_smiles)



    # TODO: we can have low rank update, priors etc...