import time
import numpy as np
import ray
from rdkit import Chem
from rdkit.Chem import QED
from LambdaZero.chem import DockVina_smi


@ray.remote(num_gpus=0)
class DockingEstimator(DockVina_smi):
    def __init__(self, dockVina_config):
        DockVina_smi.__init__(self, **dockVina_config)

    def eval(self, smiles):
        try:
            mol_name, dockscore, coord = self.dock(smiles)
        except Exception as e:
            dockscore = 0.0
        return dockscore

class DockingOracle:
    def __init__(self, num_threads, dockVina_config, mean, std):
        self.num_threads = num_threads
        # create actor pool
        self.actors = [DockingEstimator.remote(dockVina_config) for i in range(self.num_threads)]
        self.pool = ray.util.ActorPool(self.actors)
        self.mean = mean
        self.std = std
        # todo: report statistics

    def __call__(self, data):
        smiles = [d["smiles"] for d in data]
        dockscores = list(self.pool.map(lambda actor, smi: actor.eval.remote(smi), smiles))
        dockscores = [(self.mean -d) / self.std for d in dockscores]
        print("dosckscores", dockscores)
        return dockscores





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

class QEDOracle:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # create actor pool
        self.actors = [QEDEstimator.remote() for i in range(self.num_threads)]
        self.pool = ray.util.ActorPool(self.actors)

    def __call__(self, data):
        smiles = [d["smiles"] for d in data]
        qeds = list(self.pool.map(lambda actor, smi: actor.eval.remote(smi), smiles))
        print("QEDs", qeds)
        return qeds


