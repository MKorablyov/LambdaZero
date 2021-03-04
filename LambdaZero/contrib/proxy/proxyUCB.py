import numpy as np
import ray
from LambdaZero.contrib.acquisition_function import UCB
from .proxy import Proxy

@ray.remote(num_gpus=0.3, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self,update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config, logger):
        # load data for (1) acquisition function (2) proxy
        self.seen_x, self.seen_y, self.val_x, self.val_y = load_seen(**load_seen_config)
        proposed_x, proposed_d, proposed_acq = [], [], []  # todo: load if needed

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger)
        self.UCB = UCB(**acquirer_config)
        self.UCB.update_with_seen(self.seen_x, self.seen_y)
        self.oracle = oracle(**oracle_config)

    def acquire_and_update(self):
        x, d, acq, info = self.UCB.acquire_batch(self.proposed_x, self.proposed_d, self.proposed_acq)
        y = self.oracle(x)
        self.logger.log.remote([{
            "proxy/acq_mean": np.mean(self.proposed_acq), "proxy/acq_max": np.max(self.proposed_acq),
            "proxy/acquired_acq_mean": np.mean(acq), "proxy/acquired_acq_max":np.max(acq),
            "proxy/acquired_y_mean": np.mean(y), "proxy/acquired_y_max": np.max(y)
        }])
        self.seen_x.extend(x)
        self.seen_y.extend(y)
        self.UCB.update_with_seen(self.seen_x, self.seen_y) # todo: evaluate on newly acquired data
        return None

    def get_acquisition_func(self):
        return self.UCB
