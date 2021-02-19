import ray
from LambdaZero.contrib.acquisition_function import EI
from .proxy import Proxy


@ray.remote(num_gpus=0.3, num_cpus=2)
class ProxyEI(Proxy):
    def __init__(self,update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config):
        # load data for (1) acquisition function (2) proxy
        self.seen_x, self.seen_y, self.val_x, self.val_y = load_seen(**load_seen_config)
        proposed_x, proposed_d, proposed_acq = [], [], []  # todo: load if needed

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq)
        self.EI = EI(**acquirer_config)
        self.EI.update_with_seen(self.seen_x, self.seen_y)
        self.oracle = oracle(**oracle_config)

    def acquire_and_update(self):
        x, d, acq = self.EI.acquire_batch(self.proposed_x, self.proposed_d, self.proposed_acq)
        y = self.oracle(x)
        if max(y) > self.EI.best_f:
            self.EI.best_f = max(y)
            print('Best f found: {}'.format(self.EI.best_f))
        self.seen_x.extend(x)
        self.seen_y.extend(y)
        self.EI.update_with_seen(self.seen_x, self.seen_y) # todo: evaluate on newly acquired data
        return None

    def get_acquisition_func(self):
        # fixme -- I need to find a way to make a cheap copy of "acquire function"
        return self.EI
