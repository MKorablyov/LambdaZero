import os

import numpy as np
# from LambdaZero.utils.utils_op import pearson_correlation, log_likelihood
import matplotlib.pyplot as plt
import scipy

# def uncertainty_metrics(y, y_hat, y_hat_var):
#     mae = np.abs(y - y_hat).mean()
#     mse = (np.abs(y - y_hat)**2).mean()
#     log_lik = -0.5 * np.mean(np.log(2 * float(np.pi) * y_hat_var) + ((y - y_hat) ** 2 / y_hat_var))
#     return {"mae":mae, "mse":mse, "neg_log_lik":log_lik}

def uncertainty_metrics(e, e_hat):
    pear_corr = np.sum((e - np.mean(e)) * (e_hat - np.mean(e_hat))) / (
            np.sqrt(np.sum((e - np.mean(e)) ** 2) + 1e-6) * np.sqrt(np.sum((e_hat - np.mean(e_hat)) ** 2) + 1e-6))
    spearman_corr = scipy.stats.spearmanr(e, e_hat).correlation
    log_lik = -0.5 * np.mean(np.log(2 * float(np.pi) * e_hat) + (e / e_hat))
    return {"pear_corr":pear_corr,  "spearman_corr":spearman_corr, "neg_log_lik":log_lik}

def scatter_plot(x, y, labels):
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel(labels['x'])
    plt.ylabel(labels['y'])
    plt.ylim(np.min(x), np.max(x))
    save_dir = f'/home/nekoeiha/tmp/deup_2fold_{labels["features"]}/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(f'{save_dir}val_{labels["method"]}_calibrate_{labels["epochs"]}.png')

