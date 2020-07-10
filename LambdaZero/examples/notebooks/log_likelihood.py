import numpy as np
import pandas as pd
# make dataset

mu = []
sigma = []
x = []

for i in range(10000):
    sample_mu = np.random.normal(20)
    sample_sigma = np.random.uniform(low=0.0, high=3.0)

    sample_x = sample_mu + np.random.normal(scale=sample_sigma)

    mu.append(sample_mu)
    sigma.append(sample_sigma)
    x.append(sample_x)

sigma = np.asarray(sigma)
x = np.asarray(x)
mu = np.asarray(mu)

# PDF = (1 / sigma * sqrt(2 pi))  *  exp -0.5 ((x-mu)/sigma)**2
# # log PDF = - log(sigma * sqrt(2 pi)) -0.5 * ((x-mu)/sigma)**2


def logP(mu, sigma, x):
    """
    Estimate log likelihood of an estimator
    :param mu: estimated mu
    :param sigma: estimated sigma
    :param x: ground truth
    :return:
    """

    return (-np.log(sigma * (2 * np.pi)**0.5) - 0.5 * (((x - mu) / sigma) **2))

print(logP(mu, sigma,x).mean())
print(logP(mu, sigma * 10, x).mean())
print(logP(mu, sigma * 0.1,x).mean())
print(logP(mu, np.random.permutation(sigma),x).mean())