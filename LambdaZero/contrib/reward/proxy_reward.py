import time
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random

class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, **kwargs):
        self.actor = Actor(scoreProxy, actor_sync_freq)

    def reset(self, previous_reward=0.0):
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        synth_score = 1.0
        qed = 1.0
        # todo: to come up with some molecule encoding
        smiles = Chem.MolToSmiles(molecule.mol)
        dock_score = self.actor([{"smiles":smiles, "mol_graph":molecule.graph}], [qed * synth_score])[0]
        scores = {"dock_score": dock_score, "synth_score": synth_score, "qed":0.9}
        return synth_score * dock_score * qed, scores



class DummyReward:
    def __init__(self, **kwargs):
        pass

    def reset(self, previous_reward=0.0):
        self.previous_reward = 0.0
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        print(molecule, float(random()))

        return float(random()), {"reward": 1.0, "discounted_reward": 1.0, "QED": 1.0, "discount": 1.0}