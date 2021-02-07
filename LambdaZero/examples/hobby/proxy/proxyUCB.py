import ray
from LambdaZero.examples.hobby.acquisition_function import UCB,acquisition_values_UCB
from .proxy import Actor



@ray.remote(num_gpus=0.25, num_cpus=2)
class ProxyUCB(UCB): # todo: inherit Proxy()
    "combine scores from many models with uncertainty"
    def __init__(self, update_freq, proc_func, acquirer_config):
        self.update_freq = update_freq
        self.proc_func = proc_func
        self.proposed_molecules = []
        self.proposed_scores = []
        UCB.__init__(self,acquirer_config)

    def propose_molecule(self, molecule, scores):
        self.proposed_molecules.append(molecule)
        self.proposed_scores.append(scores)

        if len(self.proposed_molecules) == self.update_freq:
            self.acquire_and_update()
        print(len(self.proposed_molecules))
        return None

    def acquire_and_update(self):
        print("updating proxy", len(self.proposed_molecules))
        # graphs = self.proc_func(molecules)
        # idx = self.acquire_batch(graphs, aq_scores, discounts)
        # molecules[idx], discounts[idx]
        # dockscores = self.acquire(molecules)
        # self.update_with_seen(molecules, dockscores)
        #self.dockScore.acquire_and_update(self.proposed_molecules, self.proposed_scores)
        self.proposed_molecules, proposed_scores = [], []

    def acquire(self, molecules):
        # do docking here
        # return molecules, dockscores
        pass

    def get_weights(self):
        pass

    def get_actor(self):
        return Actor(self.proc_func,acquisition_values_UCB,self.get_weights,self.propose_molecule)
        pass


