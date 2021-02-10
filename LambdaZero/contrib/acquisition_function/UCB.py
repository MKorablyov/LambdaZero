import time
import numpy as np
from .acquisition_function import AcquisitionFunction


class UCB(AcquisitionFunction):
    def __init__(self, model, model_config, acq_size, kappa, seen_x, seen_y, val_x, val_y):

        self.kappa = kappa
        self.seen_x, self.seen_y, self.val_x, self.val_y = seen_x, seen_y, val_x, val_y
        #print("loaded training examples", len(self.seen_x), len(self.seen_y), len(self.val_x), len(self.val_y))
        AcquisitionFunction.__init__(self, model, model_config, acq_size)

    def update_with_seen(self, x, y):
        self.seen_x.extend(x)
        self.seen_y.extend(y)
        self.model.fit(x,y)
        # todo: evaluate on a separate train/val set
        return None

    def acquisition_value(self, x):
        mean, var = self.model.get_mean_and_variance(x)
        acq = [mean[i] + self.kappa * var[i] for i in range(len(mean))]
        return acq

    def acquire_batch(self, x, d, acq=None):
        if acq is not None: acq = self.acquisition_value(x)
        # compute indices with highest acquisition values
        idx = np.argsort(acq)[-self.acq_size:]
        # take selected indices
        x_ = [x[i] for i in idx]
        d_ = [d[i] for i in idx]
        acq_ = [acq[i] for i in idx]
        return x_, d_, acq_