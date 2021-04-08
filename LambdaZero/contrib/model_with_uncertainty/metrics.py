import numpy as np

def uncertainty_metrics(y, y_hat, y_hat_var):
    mae = np.abs(y - y_hat).mean()
    mse = (np.abs(y - y_hat)**2).mean()
    log_lik = -0.5 * np.mean(np.log(2 * float(np.pi) * y_hat_var) + ((y - y_hat) ** 2 / y_hat_var))
    return {"mae":mae, "mse":mse, "neg_log_lik":log_lik}
