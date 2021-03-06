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
        acq = [mean[i] + self.kappa * var[i] for i in range(len(mean))]
        info = {"mean":mean,"var":var}
        return acq, info

    def acquire_batch(self, x, d, acq=None):
        if acq is not None: acq,_ = self.acquisition_value(x)

        d = np.asarray(d)
        d[np.where(np.array(acq) < 0)[0]] = 1 # todo: discount is 0 whenever the acquisition function is negative;
                                              # there could be a better algebraic solution
        # to come up with some arithmetic way to do this

        # compute indices with highest acquisition values
        idx = np.argsort(np.array(acq) * d)[-self.acq_size:]
        # take selected indices
        x_acquired = [x[i] for i in idx]
        d_acquired= [d[i] for i in idx]
        acq_acquired = [acq[i] for i in idx]
        info = {}
        return x_acquired, d_acquired, acq_acquired, info