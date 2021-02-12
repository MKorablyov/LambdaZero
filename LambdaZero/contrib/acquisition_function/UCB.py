import time
import numpy as np
from .acquisition_function import AcquisitionFunction


class UCB(AcquisitionFunction):
    def __init__(self, model, model_config, acq_size, kappa):
        self.kappa = kappa
        self.model = model(**model_config)
        AcquisitionFunction.__init__(self, model, model_config, acq_size)

    def update_with_seen(self, x, y):
        self.model.fit(x,y)
        # todo: evaluate acquisition function on a separate train/val set
        return None

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