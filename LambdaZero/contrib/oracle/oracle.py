import time
import numpy as np
import ray
from rdkit import Chem
from rdkit.Chem import QED
from LambdaZero.chem import DockVina_smi
from LambdaZero.models import ChempropWrapper_v1


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
    def __init__(self, num_threads, dockVina_config, mean, std, logger):
        self.num_threads = num_threads
        # create actor pool
        self.actors = [DockingEstimator.remote(dockVina_config) for i in range(self.num_threads)]
        self.pool = ray.util.ActorPool(self.actors)
        self.mean = mean
        self.std = std
        self.logger = logger
        # todo: report statistics

    def __call__(self, data):
        smiles = [d["smiles"] for d in data]
        dockscores = list(self.pool.map(lambda actor, smi: actor.eval.remote(smi), smiles))
        self.logger.log.remote({"docking_oracle/raw_dockscore_mean": np.mean(dockscores),
                                "docking_oracle/raw_dockscore_min": np.min(dockscores)})
        dockscores = [min(0, d) for d in dockscores] # when docking score is positive, clip at 0
        dockscores = [(self.mean -d) / self.std for d in dockscores] # this normalizes and flips dockscore
        self.logger.log.remote({"docking_oracle/norm_dockscore_mean": np.mean(dockscores),
                                "docking_oracle/norm_dockscore_max": np.max(dockscores)})
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

    def __call__(self, molecules):
        smiles = [m["smiles"] for m in molecules]
        qeds = list(self.pool.map(lambda actor, smi: actor.eval.remote(smi), smiles))
        return qeds


@ray.remote(num_gpus=0.05)
class ChempropWrapper_v2(ChempropWrapper_v1):
    def eval(self, m):
        #print("M:", m)
        return ChempropWrapper_v1.__call__(self, m)

class SynthOracle:
    def __init__(self, synth_options, synth_config):
        self.synth_net = ChempropWrapper_v2.options(**synth_options).remote(synth_config)

    def __call__(self,molecules):
        synths = []
        for m in molecules:
            try:
                synth = ray.get(self.synth_net.eval.remote(m["mol"]))
            except Exception as e:
                print(e)
                synth = 0.0
            synths.append(synth)
        return synths


