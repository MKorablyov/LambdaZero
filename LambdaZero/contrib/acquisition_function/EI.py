import time
import numpy as np
import torch
from .acquisition_function import AcquisitionFunction
from torch.distributions import Normal

class EI(AcquisitionFunction):
    def __init__(self, model, model_config, acq_size, best_f=0):
        self.model = model(**model_config)
        self.best_f = best_f
        AcquisitionFunction.__init__(self, model, model_config, acq_size)

    def update_with_seen(self, x, y):
        self.model.fit(x,y)
        # todo: evaluate acquisition function on a separate train/val set
        return None

    def acquisition_value(self, x):
        mean, var = self.model.get_mean_and_variance(x)
        mean = torch.tensor(mean)
        var = torch.tensor(var)

        sigma = var.clamp_min(1e-9).sqrt()
        u = (mean - self.best_f) / sigma
        normal = Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        acq = sigma * (updf + u * ucdf)
        import pdb;
        pdb.set_trace()
        return np.array(acq)

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