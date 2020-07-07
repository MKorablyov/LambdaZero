
import torch
from torch import nn
from torch import distributions
from sklearn import linear_model
import numpy as np

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

        # We'll get Sklearn's version of this class to compute the best hyperparamters for us.
        # Internally this currently works in a  compute efficient manner doing SVD, which could get ported to
        # PyTorch but dont think any point currently...
        clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False, verbose=True)
        clf.fit(x_train.detach().numpy(), y_train.detach().squeeze().numpy())
        self.alpha = clf.lambda_
        self.beta = clf.alpha_
        # ^ sorry we are following the names in Snoek et al so our alpha is unfortunately not same as sklearns...
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

    def predict(self, x_test, use_noise=True):
        """
        :param x_test: [N_test, M]
        """
        n_test = x_test.shape[0]
        mean = x_test @ self.m + self.mean_ # [N_test, 1]

        chol_inv_x_tran, _ = torch.triangular_solve(x_test.T, self.chol_K, upper=False)
        variance = chol_inv_x_tran.T @ chol_inv_x_tran
        if use_noise:
            variance += 1./self.beta * torch.eye(n_test)  # [N_test, N_test]
        variance += 1e-9 * torch.eye(n_test)  # <- this is for numerical stability.
        out = distributions.MultivariateNormal(mean.squeeze(1), variance)

        return out

    def get_predict_mean_and_variance_func(self):
        """
        returns a function
        that should take in X_test (eg this could be a tensor of dim) [N, d]
        and return a two element tuple of tensor ([N,1], [N,1])
        which contains the predictive mean and (marginal) variance at that output point
        f is the dimensionality of y, which in most cases is probably 1.
        N is the batch size
        """
        def func(x_test_batch):
            mean = x_test_batch @ self.m + self.mean_  # [N_test, 1]

            chol_inv_x_tran, _ = torch.triangular_solve(x_test_batch.T, self.chol_K, upper=False)
            marginal_variance = torch.sum(chol_inv_x_tran*chol_inv_x_tran,dim=0) + 1./self.beta  # [N_test]
            marginal_variance = marginal_variance[:, None]  #[N_test, 1]
            return mean, marginal_variance
        return func

    def get_predict_mean_and_variance_func_for_smiles(self):
        func = self.get_predict_mean_and_variance_func()

        def new_func(x_test_smiles_batch):
            feats = self._convert_smiles_to_features(x_test_smiles_batch)
            return func(feats)
        return new_func

    def get_sampler_func(self, use_noise=True):
        """
        Samples from the uncertainty over the parameters and returns a deterministic model that returns the samples
        at each point.
        nb ignoring the noise parametrer
        """
        weight_dist = distributions.MultivariateNormal(self.m.squeeze(1), precision_matrix=self.K)
        sampled_weights = weight_dist.rsample()[:, None]  # [feat_size, 1]
        if sampled_weights.shape != (self.m.shape[0], 1):
            print("sampled weights shape", sampled_weights.shape)
            print("m shape", self.m.shape)
            raise RuntimeError

        def sample_func(x_test_batch):
            out = x_test_batch @ sampled_weights + self.mean_  # [N_test, 1]
            if use_noise:
                out += np.sqrt(1./self.beta) * torch.randn(x_test_batch.shape[0], 1)  # [N_test, 1]
            return out
        return sample_func

    def get_sampler_func_for_smiles(self, use_noise=True):
        func = self.get_sampler_func(use_noise)

        def new_func(x_test_smiles_batch):
            feats = self._convert_smiles_to_features(x_test_smiles_batch)
            return func(feats)
        return new_func

    def predict_on_smiles(self, x_test_smiles):
        feats = self._convert_smiles_to_features(x_test_smiles)
        return self.predict(feats)

    def _convert_smiles_to_features(self, list_of_smiles):
        return self.embedding_func(list_of_smiles)

    #TODO: we can have low rank update, sample from priors