import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config

import ray

def _satlins(x, cutoff0, cutoff1):
    "shifted saturated linearity activation function _/-"
    x = (x - cutoff0) / (cutoff1 - cutoff0)
    x = min(max(0.0, x), 1.0)  # relu to maxout at 1
    return x

class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff, clip_dockreward, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.qed_oracle = QEDOracle(num_threads=1)
        self.synth_oracle = SynthOracle(synth_config)
        self.dockProxy_actor = Actor(scoreProxy, actor_sync_freq)
        self.clip_dockreward = clip_dockreward

        self.rewards = []

    def reset(self, previous_reward=0.0):
        return None

    def eval(self, molecule):
        smiles = Chem.MolToSmiles(molecule.mol)
        qed = self.qed_oracle([{"smiles":smiles, "mol":molecule.mol}])[0]
        synth_score = self.synth_oracle([{"smiles":smiles, "mol":molecule.mol}])[0]
        # stop optimizing qed/synth beyond thresholds
        clip_qed = _satlins(qed, self.qed_cutoff[0], self.qed_cutoff[1])
        clip_synth = _satlins(synth_score, self.synth_cutoff[0], self.synth_cutoff[1])

        dockreward = self.dockProxy_actor([{"smiles":smiles, "mol_graph":molecule.graph, "env_name": self.env_name}],
                                          [clip_qed * clip_synth])[0]
        self.rewards.append(dockreward)
        if self.clip_dockreward is not None: # rewards are clipped to be non-negative (there is not anything to avoid)
            # I had to add an artificial reward for any molecule to help this to converge
            clip_dock = max(self.clip_dockreward + dockreward, 0.0)
        else:
            clip_dock = dockreward

        info = {"dockreward": dockreward, "synth_score": synth_score, "qed":qed}

        return clip_qed * clip_synth * clip_dock, info

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        return self.eval(molecule)



class ProxyRewardSparse(ProxyReward):
    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        if agent_stop or env_stop:
            reward, info = ProxyReward.eval(self, molecule)
        else:
            reward, info = 0.0, {}
        return reward, info


class DummyReward:
    def __init__(self, **kwargs):
        pass

    def reset(self, previous_reward=0.0):
        self.previous_reward = 0.0
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        print(molecule, float(random()))

        return float(random()), {"reward": 1.0, "discounted_reward": 1.0, "QED": 1.0, "discount": 1.0}