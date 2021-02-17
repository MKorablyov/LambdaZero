import time
import numpy as np
import ray
from rdkit import Chem
from rdkit.Chem import QED



@ray.remote
class QEDEstimator:
    def __init__(self):
        pass

    def eval(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
        except Exception as e:
            qed = 0.0
        return qed

class DockingOracle:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # create actor pool
        self.actors = [QEDEstimator.remote() for i in range(self.num_threads)]
        self.pool = ray.util.ActorPool(self.actors)

        # todo: normalization
        # todo: bounds/discounts
        # todo: report statistics

    def __call__(self, data):
        smiles = [d["smiles"] for d in data]
        qeds = np.array(list(self.pool.map(lambda actor, smi: actor.eval.remote(smi), smiles)))
        print("QEDs", qeds)
        return qeds



# class DockingOracle:
#     def __init__(self, num_threads):
#         self.num_threads = num_threads
#         # create actor pool
#
#     def __call__(self, data):
#
#         dockscores = [1.0 for i in range(len(data))]
#
#         print("oracle is called on", len(data), dockscores)
#         return dockscores