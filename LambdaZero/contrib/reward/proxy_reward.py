import time
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem

class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, **kwargs):
        self.actor = Actor(scoreProxy, actor_sync_freq)

    def reset(self):
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        synth_score = 0.5
        qed = 0.9
        # todo: to come up with some molecule encoding that could be sent over ray
        smiles = Chem.MolToSmiles(molecule.mol)
        dock_score = self.actor({"smiles":smiles, "mol_graph":molecule.graph}, [qed * synth_score])[0]
        scores = {"dock_score":dock_score, "synth_score": synth_score, "qed":0.9}
        return synth_score * dock_score * qed, scores

