import os
from rdkit import Chem
from rdkit.Chem import QED
from guacamol.scoring_function import ScoringFunctionBasedOnRdkitMol, BatchScoringFunction
from typing import List
import LambdaZero.chem
import LambdaZero.utils

from LambdaZero.environments.block_mol_graph_v1 import GraphMolObs
from LambdaZero.environments.molMDP import BlockMoleculeData

_, _, summaries_dir = LambdaZero.utils.get_external_dirs()

default_config = {
    "env_eval_config": {
        "dockscore_norm": [-8.6, 1.10],
        "qed_cutoff": [0., 0.5],
    }
}

config = default_config
out_dir = os.path.join(summaries_dir, "guacamol_oracle_dock_eval")


class BlockMoleculeData_wrapper:
    def __init__(self, mol, smiles, graph=None):
        self.graph = graph
        self.mol = mol
        self.smiles = smiles


class EnvEval:
    def __init__(self, config):
        self.dock_smi = LambdaZero.chem.DockVina_smi(outpath=out_dir)
        self.dockscore_norm = config["dockscore_norm"]
        self.qed_cutoff = config["qed_cutoff"]

    def __call__(self, smi, docking=True):
        if docking:
            try:
                # Since synthesizability prediction varies with training dataset, we ignore it here.
                # smi = Chem.MolToSmiles(mol)
                gridscore = self.dock_smi.dock(smi)[1][0]
                dock_reward = -((gridscore - self.dockscore_norm[0]) / self.dockscore_norm[1])

                mol = Chem.MolFromSmiles(smi)
                qed = QED.qed(mol)
                qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
                qed_discount = min(max(0.0, qed_discount), 1.0)  # relu to maxout at 1

                reward = dock_reward * qed_discount
            except Exception as e:
                print('exception for', e)
                dock_reward = None
                reward = 0.
            print(reward)
            return reward  # dock_reward


class Oracle_wrapper(BatchScoringFunction):
    def __init__(self, config) -> None:
        super().__init__()

        self.oracle_reward = config["reward"](config["reward_config"])

    def raw_score_list(self, smiles_list: List[str]) -> List[float]:
        rewards = []
        for smiles in smiles_list:
            reward = self.oracle_reward(smiles)
            rewards.append(reward)
        return rewards

class Proxy_wrapper(BatchScoringFunction):

    def __init__(self, config) -> None:
        super().__init__()

        # initialize scoreProxy which would be shared across many agents
        self.scoreProxy = config['reward_config']['scoreProxy']. \
            options(**config['reward_config']['scoreProxy_options']). \
            remote(**config['reward_config']['scoreProxy_config'])
        config['reward_config']['scoreProxy'] = self.scoreProxy

        # initialize proxy reward
        self.proxy_reward = config['reward'](**config['reward_config'])

        # initialize mol to graph
        self.graphmolobs = GraphMolObs()

    def score(self, smiles) -> float:
        mol = Chem.MolFromSmiles(smiles)
        molecule = BlockMoleculeData_wrapper(mol, smiles)
        try:
            mol_blockmoldata = BlockMoleculeData()
            mol_blockmoldata._mol = mol
            graph, _ = self.graphmolobs(mol_blockmoldata)
            molecule.graph = graph
            reward, log_vals = self.proxy_reward(molecule, True, True, None)  # always evaluate
        except Exception as e:
            print(e)
            reward = 0.0
            log_vals = {}
        return reward

    def raw_score_list(self, smiles_list: List[str]) -> List[float]:
        rewards = []
        for smiles in smiles_list:
            reward = self.score(smiles)
            rewards.append(reward)
        return rewards

# class Proxy_wrapper(BatchScoringFunction):
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         self.env_eval = EnvEval(config["env_eval_config"])
#
#     def raw_score_list(self, smiles_list: List[str]) -> List[float]:
#         rewards = []
#         for smiles in smiles_list:
#             reward = self.env_eval(smiles)
#             rewards.append(reward)
#         return rewards