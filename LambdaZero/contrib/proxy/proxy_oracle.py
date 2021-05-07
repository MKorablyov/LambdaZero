import ray
from .proxy import Proxy


@ray.remote(num_gpus=0.0, num_cpus=2)
class ProxyPredockedOnly(Proxy):
    def __init__(self, update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config, logger):
        proposed_x, proposed_d, proposed_acq = [], [], []

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger)
        self.oracle = oracle(**oracle_config)

        self.oracle_config = oracle_config

    def get_oracle_config(self):
        return self.oracle_config

    def get_acquisition_func(self):
        return None

    def update_predocked(self, smiles: str, dockscore: float):
        if dockscore is None:
            # Update failed hits
            if smiles in self.oracle._failed_hits:
                self.oracle._failed_hits[smiles] += 1
            else:
                self.oracle._failed_hits[smiles] = 1
        else:
            self.oracle.predocked.local_update(smiles, dockscore)

    def propose_x(self,x, d, acq):
        return None

    def get_oracle(self, x):
        return self.oracle(x)

    def set_oracle_fix_none(self, vl):
        self.oracle.set_fix_none(vl)

    def set_oracle_norm_data(self, vl):
        self.oracle.set_norm_data(vl)


@ray.remote(num_gpus=0.0, num_cpus=2)
class ProxyDummy(Proxy):
    def __init__(self, update_freq, acquirer_config, oracle, oracle_config, load_seen, load_seen_config, logger):
        proposed_x, proposed_d, proposed_acq = [], [], []

        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger)
        # self.oracle = oracle(**oracle_config)

        self.oracle_config = oracle_config

    def get_oracle_config(self):
        return self.oracle_config

    def get_acquisition_func(self):
        return None

    def update_predocked(self, smiles: str, dockscore: float):
        pass

    def propose_x(self, x, d, acq):
        return None

    def get_oracle(self, x):
        return None

    def set_oracle_fix_none(self, vl):
        pass

    def set_oracle_norm_data(self, vl):
        pass