import ray
from LambdaZero.contrib.acquisition_function import MES
from .proxy import Proxy


@ray.remote(num_gpus=0.3, num_cpus=2)
class ProxyMES(Proxy):
    def __init__(self,update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config):
        # load data for (1) acquisition function (2) proxy
        self.seen_x, self.seen_y, self.val_x, self.val_y = load_seen(**load_seen_config)
        proposed_x, proposed_d, proposed_acq = [], [], []  # todo: load if needed

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq)
        self.MES = MES(**acquirer_config)
        self.MES.gp = self.MES.update_with_seen(self.seen_x, self.seen_y)
        self.oracle = oracle(**oracle_config)

    def acquire_and_update(self):
        x, d, acq = self.MES.acquire_batch(self.proposed_x, self.proposed_d, self.proposed_acq)
        y = self.oracle(x)

        self.seen_x.extend(x)
        self.seen_y.extend(y)
        self.MES.update_with_seen(self.seen_x, self.seen_y) # todo: evaluate on newly acquired data
        return None

    def get_acquisition_func(self):
        # fixme -- I need to find a way to make a cheap copy of "acquire function"
        return self.MES
