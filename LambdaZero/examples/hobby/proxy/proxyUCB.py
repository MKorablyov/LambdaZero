import ray
from LambdaZero.examples.hobby.acquisition_function import UCB
from .proxy import Proxy, Actor


@ray.remote(num_gpus=0.25, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, sync_freq, proc_func, acquirer_config,
                 proposed_x=[],proposed_acq=[],proposed_d=[]):

        self.proc_func = proc_func
        self.UCB = UCB(acquirer_config)
        Proxy.__init__(self, sync_freq, proposed_x, proposed_acq, proposed_d)

    def acquire_and_update(self):
        print("updating proxy", len(self.proposed_x))
        # graphs = self.proc_func(molecules)
        # idx = self.acquire_batch(graphs, aq_scores, discounts)
        # molecules[idx], discounts[idx]
        # dockscores = self.acquire(molecules)
        # self.update_with_seen(molecules, dockscores)
        # self.dockScore.acquire_and_update(self.proposed_molecules, self.proposed_scores)

    def acquire(self, x):
        # do docking here
        # return molecules, dockscores
        pass

    def get_acquisition_func(self):
        return self.UCB
