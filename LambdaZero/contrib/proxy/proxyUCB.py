import ray
from LambdaZero.contrib.acquisition_function import UCB
from .proxy import Proxy


@ray.remote(num_gpus=0.25, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, update_freq, acquirer_config, oracle, oracle_config,
                 proposed_x=[], proposed_d=[], proposed_acq=[]):
        self.UCB = UCB(acquirer_config)
        self.oracle = oracle(**oracle_config)
        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq)

    def acquire_and_update(self):
        print("updating proxy", len(self.proposed_x))
        x, d, acq = self.UCB.acquire_batch(self.proposed_x, self.proposed_d, self.proposed_acq)
        y = self.oracle(x)
        self.UCB.update_with_seen(x,y)
        return None

    def get_acquisition_func(self):
        return self.UCB
