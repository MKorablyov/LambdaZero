import ray
from LambdaZero.examples.hobby.acquisition_function import UCB
from .proxy import Proxy, Actor


@ray.remote(num_gpus=0.25, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, update_freq, acquirer_config, proposed_x=[], proposed_d=[], proposed_acq=[]):
        self.UCB = UCB(acquirer_config)
        Proxy.__init__(self, update_freq, proposed_x, proposed_d, proposed_acq)

    def acquire_and_update(self):
        print("updating proxy", len(self.proposed_x))

        x, d, acq = self.UCB.acquire_batch(self.proposed_x, self.proposed_d, self.proposed_acq)

        # dockscores = self.acquire(molecules)
        # self.update_with_seen(molecules, dockscores)
        # self.dockScore.acquire_and_update(self.proposed_molecules, self.proposed_scores)

    def acquire(self, x):

        # todo: I need to decide how to acquire;
        # one of the possibilities is to have some expensive class to run


        # do docking here
        # return molecules, dockscores
        pass

    def get_acquisition_func(self):
        return self.UCB
