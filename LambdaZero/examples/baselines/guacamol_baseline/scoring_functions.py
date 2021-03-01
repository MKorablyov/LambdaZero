import os.path as osp
from rdkit import Chem
from rdkit.Chem import QED
from guacamol.scoring_function import ScoringFunctionBasedOnRdkitMol, BatchScoringFunction
from typing import List

import LambdaZero.chem
import LambdaZero.utils

default_config = {
    "env_eval_config": {
        "dockscore_norm": [-8.6, 1.10],
        "qed_cutoff": [0., 0.5],
    }
}

config = default_config
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
out_dir = osp.join(summaries_dir, "lstm_dock_eval")
mgltools_dir=osp.join(programs_dir, "mgltools_x86_64Linux2_1.5.6")
vina_dir=osp.join(programs_dir, "vina")
docksetup_dir=osp.join(datasets_dir, "seh/4jnc")

class EnvEval:
    def __init__(self, config):
        self.dock_smi = LambdaZero.chem.DockVina_smi(outpath=out_dir,
                                                     mgltools_dir=mgltools_dir,
                                                     vina_dir=vina_dir,
                                                     docksetup_dir=docksetup_dir)
        self.dockscore_norm = config["dockscore_norm"]
        self.qed_cutoff = config["qed_cutoff"]

    def __call__(self, smi, docking=True):
        if docking:
            try:
                # Since synthesizability prediction varies with training dataset, we ignore it here.
                # smi = Chem.MolToSmiles(mol) #
                gridscore = self.dock_smi.dock(smi)[1]
                dock_reward = -((gridscore - self.dockscore_norm[0]) / self.dockscore_norm[1])

                mol = Chem.MolFromSmiles(smi) #
                qed = QED.qed(mol)
                qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
                qed_discount = min(max(0.0, qed_discount), 1.0)  # relu to maxout at 1

                reward = dock_reward * qed_discount
            except Exception as e:
                print('exception for', e)
                dock_reward = None
                reward = 0.
            print (reward)
            return reward # dock_reward

# class Oracle_wrapper(ScoringFunctionBasedOnRdkitMol):
#
#     def __init__(self) -> None:
#         super().__init__()
#
#         self.env_eval = EnvEval(config["env_eval_config"])
#
#     def score_mol(self, mol: Chem.Mol) -> float:
#         reward = self.env_eval(mol)
#         return reward

class Oracle_wrapper(BatchScoringFunction):

    def __init__(self) -> None:
        super().__init__()

        self.env_eval = EnvEval(config["env_eval_config"])

    def raw_score_list(self, smiles_list: List[str]) -> List[float]:
        rewards = []
        for smiles in smiles_list:
            reward = self.env_eval(smiles)
            rewards.append(reward)
        return rewards


class Proxy_wrapper(ScoringFunctionBasedOnRdkitMol):

    def __init__(self, reward, config) -> None:
        super().__init__()

        self.env_eval = reward(**config["reward_config"])

    def score_mol(self, mol: Chem.Mol) -> float:
        reward = self.env_eval(mol)
        return reward

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