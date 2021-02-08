import time
import numpy as np
from .acquisition_function import AcquisitionFunction


class UCB(AcquisitionFunction):
    def __init__(self, config):
        AcquisitionFunction.__init__(self, config["model"], config["model_config"], config["acq_size"])
        self.kappa = config["kappa"]

    def update_with_seen(self, x, y):
        # self.seen_x +=x
        # self.model_with_uncertainty.fit(x,y, self.val_x, self.val_y)
        pass

    def acquisition_value(self, x):
        mean, var = self.model.get_mean_and_variance(x)
        acq = [mean[i] + self.kappa * var[i] for i in range(len(mean))]
        return acq

    def acquire_batch(self, x, d, acq=None):
        if acq is not None:
            acq = self.acquisition_value(x)

        # compute indices with highest acquisition values
        idx = np.argsort(acq)[-self.acq_size:]
        # take selected indices
        x_ = [x[i] for i in idx]
        d_ = [d[i] for i in idx]
        acq_ = [acq[i] for i in idx]
        return x_, d_, acq_