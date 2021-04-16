import time
import numpy as np
from .acquisition_function import AcquisitionFunction


class UCB(AcquisitionFunction):
    def __init__(self, model, model_config, acq_size, kappa):
        self.kappa = kappa
        self.model = model(**model_config)
        AcquisitionFunction.__init__(self, model, model_config, acq_size)

    def update_with_seen(self, x, y, x_new, y_new):
        self.model.update(x, y, x_new, y_new)
        return None

    def acquisition_value(self, x):
        mean, var = self.model.get_mean_and_variance(x)
        acq = [mean[i] + self.kappa * (var[i]**0.5) for i in range(len(mean))]
        info = {"mean":mean, "var":var}
        return acq, info

    def acquire_batch(self, x, d, acq=None):
        if acq is None: acq,_ = self.acquisition_value(x)
        # compute indices with highest acquisition values
        v = np.asarray(acq) * np.asarray(d)
        idx = np.argsort(v)[-self.acq_size:]
        # take selected indices
        x_acquired = [x[i] for i in idx]
        d_acquired = [d[i] for i in idx]
        acq_acquired = [acq[i] for i in idx]

        info = {"x_unacquired": [x[i] for i in range(len(x)) if i not in idx]}
        return x_acquired, d_acquired, acq_acquired, info