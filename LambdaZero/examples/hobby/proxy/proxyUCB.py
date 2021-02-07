import ray
from LambdaZero.examples.hobby.acquisition_function import UCB#,acquisition_values_UCB
from .proxy import Proxy, Actor


@ray.remote(num_gpus=0.25, num_cpus=2)
class ProxyUCB(Proxy):
    def __init__(self, update_freq, proc_func, acquirer_config,
                 proposed_x=[],proposed_y=[],proposed_d=[]):

        self.proc_func = proc_func
        self.UCB = UCB(acquirer_config)
        Proxy.__init__(self, update_freq, proposed_x, proposed_y, proposed_d)

    def acquire_and_update(self):
        print("updating proxy", len(self.proposed_x))
        # graphs = self.proc_func(molecules)
        # idx = self.acquire_batch(graphs, aq_scores, discounts)
        # molecules[idx], discounts[idx]
        # dockscores = self.acquire(molecules)
        # self.update_with_seen(molecules, dockscores)
        #self.dockScore.acquire_and_update(self.proposed_molecules, self.proposed_scores)

    def acquire(self, molecules):
        # do docking here
        # return molecules, dockscores
        pass

    def get_acquisition_func(self):
        return self.UCB

    #def get_model(self):
    #    return self.model
    #def get_actor(self):
    #    return Actor(self.proc_func,acquisition_values_UCB,self.get_weights,self.propose_x)


