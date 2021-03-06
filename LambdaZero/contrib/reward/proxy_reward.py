import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config

import ray

from LambdaZero.inputs.inputs_op import _brutal_dock_proc

def _satlins(x, cutoff0, cutoff1):
    "shifted saturated linearity activation function _/-"
    x = (x - cutoff0) / (cutoff1 - cutoff0)
    x = min(max(0.0, x), 1.0)  # relu to maxout at 1
    return x

class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff, synth_options, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.qed_oracle = QEDOracle(num_threads=1)
        self.synth_oracle = SynthOracle(synth_options, synth_config)
        self.dockProxy_actor = Actor(scoreProxy, actor_sync_freq)

    def reset(self, previous_reward=0.0):
        return None

    def eval(self, molecule):
        #print([w.detach().sum().cpu().numpy() for w in self.dockProxy_actor.acquisition_func.model.model.parameters()])

        qed = self.qed_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        synth_score = self.synth_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        # stop optimizing qed/synth beyond thresholds
        clip_qed = _satlins(qed, self.qed_cutoff[0], self.qed_cutoff[1])
        clip_synth = _satlins(synth_score, self.synth_cutoff[0], self.synth_cutoff[1])


        # todo: RL graph carries extra features which we don't have for smiles molecules
        # ideally we would like to reuse the same graphs in all model
        #graph = molecule.graph
        molecule.graph.x = molecule.graph.x[:,:14]
        print(molecule.graph.x.shape)
        proxy_dock, actor_info = self.dockProxy_actor([{"smiles":molecule.smiles, "mol_graph":molecule.graph,
                                                        "env_name": self.env_name}], [clip_qed * clip_synth])

        if np.random.uniform() > 0.99:
            try:
                #graph2 = ray.get(_brutal_dock_proc.remote(molecule.smiles, {}, None, None))
                #print("graph2", graph2)
                #time.sleep(100)
                #proxy_dock2, actor_info2 = self.dockProxy_actor([{"smiles": molecule.smiles, "mol_graph": graph2,
                #                                                "env_name": self.env_name}], [clip_qed * clip_synth])
                graph3 = ray.get(_brutal_dock_proc.remote("O=C(CN1C(=O)c2ccccc2C1=O)N1CCN(c2nnc(-c3ccccc3)c3ccccc32)CC1",
                                                          {}, None, None))
                proxy_dock3, actor_info3 = self.dockProxy_actor([{"smiles": molecule.smiles, "mol_graph": graph3,
                                                            "env_name": self.env_name}], [clip_qed * clip_synth])

                print("proxy dock", proxy_dock, "good mol", proxy_dock3)
                print(actor_info3)
            except Exception as e:
                print(e)

        proxy_dock = float(proxy_dock[0]) # actor works on multiple x by default
        if proxy_dock > 0: # reward should be rarely negative; when negative, discount won't be applied
            reward = clip_qed * clip_synth * proxy_dock
        else:
            reward = proxy_dock
        info = {"proxy_dock": proxy_dock,
                "proxy_dock_mean":actor_info["mean"][0],
                "proxy_dock_var": actor_info["var"][0],
                "synth_score": synth_score, "qed_score":qed,
                "clip_qed": clip_qed, "clip_synth": clip_synth}
        return reward, info

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
        self.qed_oracle = QEDOracle(num_threads=1)

    def reset(self, previous_reward=0.0):
        self.previous_reward = 0.0
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        return float(random()), {"reward": 1.0, "discounted_reward": 1.0, "QED": 1.0, "discount": 1.0}