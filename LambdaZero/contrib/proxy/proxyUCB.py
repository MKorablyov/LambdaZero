import random

import numpy as np
import ray
from LambdaZero.contrib.acquisition_function import UCB
from .proxy import Proxy

@ray.remote(num_gpus=0.3, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config,
                 after_acquire, logger):
        self.after_acquire = after_acquire
        # load data for (1) acquisition function (2) proxy
        self.seen_x, self.seen_y, self.val_x, self.val_y = load_seen(**load_seen_config)
        proposed_x, proposed_d, proposed_acq = [], [], []

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger)
        self.UCB = UCB(**acquirer_config)
        self.UCB.model.fit(self.seen_x, self.seen_y)
        self.oracle = oracle(**oracle_config)

    def acquire_and_update(self, proposed_x, proposed_d, proposed_acq):
        # if self.acquired_batches < self.init_random_batches:
        x, d, acq, info = self.UCB.acquire_batch(proposed_x, proposed_d, proposed_acq)
        y = self.oracle(x)
        v = np.asarray(y) * np.asarray(d)
        # perform post acquisition function(s)
        [func(proposed_x, proposed_d, proposed_acq, x, d, acq, info, y) for func in self.after_acquire]

        self.logger.log.remote([{
            "proxy/proposed_acq_mean": np.mean(proposed_acq),
            "proxy/proposed_acq_max": np.max(proposed_acq),
            "proxy/proposed_acq_min": np.min(proposed_acq),

            "proxy/acquired_acq_mean": np.mean(acq),
            "proxy/acquired_acq_max": np.max(acq),
            "proxy/acquired_acq_min": np.min(acq),

            "proxy/acquired_v_mean": np.mean(v),
            "proxy/acquired_v_max": np.max(v),
            "proxy/acquired_v_min": np.min(v)
        }])
        self.UCB.update_with_seen(self.seen_x, self.seen_y, x, y)
        self.seen_x, self.seen_y = self.seen_x + x, self.seen_y + y
        return None

    def get_acquisition_func(self):
        return self.UCB