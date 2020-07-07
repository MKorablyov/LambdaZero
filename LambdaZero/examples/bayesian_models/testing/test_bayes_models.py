
import pytest

from sklearn import linear_model
import numpy as np
import torch

from bayes_vs import bayes_models

def y_np(x):
    return 5*x + 2*x*np.sin(x)

def dummy_data(num_data_points):
    x = 5 * np.random.randn(num_data_points)[:, None]
    y = y_np(x)
    return x, y


def test_predict_1d():
    x_train, y_train = dummy_data(20)



    # We're gonna fit the model using sklearn fit model so we know exactly the same.
    clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
    clf.fit(x_train, y_train[:, 0])

    embed = lambda x: x
    embed.fp_dim = 1
    bayes_ridge = bayes_models.BayesianRegression(embed, fit_mean=False)
    bayes_ridge.alpha = clf.lambda_
    bayes_ridge.beta = clf.alpha_
    bayes_ridge.m = torch.tensor(clf.coef_)[:, None]
    assert clf.sigma_.size == 1
    bayes_ridge.K = torch.tensor(1. / clf.sigma_)
    bayes_ridge.chol_K = torch.cholesky(bayes_ridge.K)

    # We're now gonna test both the sklearn and our model on the data and make sure the predictions match
    x_test, y_test = dummy_data(5)

    y_sklearn_mn, y_sklearn_std = clf.predict(x_test, return_std=True)
    y_mvn = bayes_ridge.predict(torch.tensor(x_test))
    us_mn = y_mvn.mean.numpy()
    us_std = torch.sqrt(torch.diag(y_mvn.covariance_matrix)).numpy()

    np.testing.assert_array_almost_equal(y_sklearn_mn, us_mn)
    np.testing.assert_array_almost_equal(y_sklearn_std, us_std, decimal=3)


def test_predict_1d_batched():
    x_train, y_train = dummy_data(20)


    # We're gonna fit the model using sklearn fit model so we know exactly the same.
    clf = linear_model.BayesianRidge(compute_score=True, fit_intercept=False)
    clf.fit(x_train, y_train[:, 0])

    embed = lambda x: x
    embed.fp_dim = 1
    bayes_ridge = bayes_models.BayesianRegression(embed, fit_mean=False)
    bayes_ridge.alpha = clf.lambda_
    bayes_ridge.beta = clf.alpha_
    bayes_ridge.m = torch.tensor(clf.coef_)[:, None]
    assert clf.sigma_.size == 1
    bayes_ridge.K = torch.tensor(1. / clf.sigma_)
    bayes_ridge.chol_K = torch.cholesky(bayes_ridge.K)

    # We're now gonna test both the sklearn and our model on the data and make sure the predictions match
    x_test, y_test = dummy_data(5)

    y_sklearn_mn, y_sklearn_std = clf.predict(x_test, return_std=True)

    br_pred = bayes_ridge.get_predict_mean_and_variance_func()
    us_mn, us_var = br_pred(torch.tensor(x_test))
    us_mn = us_mn.numpy()[:, 0]
    us_std = torch.sqrt(us_var).numpy()[:,0]

    np.testing.assert_array_almost_equal(y_sklearn_mn, us_mn)
    np.testing.assert_array_almost_equal(y_sklearn_std, us_std)









