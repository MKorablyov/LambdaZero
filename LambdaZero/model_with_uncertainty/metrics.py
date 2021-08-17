import os

import numpy as np
import matplotlib.pyplot as plt
import scipy


def uncertainty_metrics(y, y_hat, y_hat_var):
    e = np.abs(y - y_hat)**2
    mse_deup = (np.abs(e - y_hat_var)**2).mean()
    log_lik = -0.5 * np.mean(np.log(2 * float(np.pi) * y_hat_var) + ((y - y_hat) ** 2 / y_hat_var))

    pear_corr = np.sum((e - np.mean(e)) * (y_hat_var - np.mean(y_hat_var))) / (
            np.sqrt(np.sum((e - np.mean(e)) ** 2) + 1e-6) * np.sqrt(np.sum((y_hat_var - np.mean(y_hat_var)) ** 2) + 1e-6))
    spearman_corr = scipy.stats.spearmanr(e, y_hat_var).correlation
    return {"mse":e.mean(), "mse_deup":mse_deup, "neg_log_lik":log_lik, "pear_corr":pear_corr,  "spearman_corr":spearman_corr}


def create_uncertainty_metrics():

    return {"mse":[], "mse_deup":[], "neg_log_lik":[], "pear_corr":[],  "spearman_corr":[]}