import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config

import LambdaZero.contrib.functional

class ProxyReward_v2:
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff, synth_options, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.qed_oracle = QEDOracle(num_threads=1)
        self.synth_oracle = SynthOracle(synth_options, synth_config)
        self.dockProxy_actor = Actor(scoreProxy, actor_sync_freq)

    def reset(self, previous_reward=0.0):
        return None

    def eval(self, traj):
        molecule = traj[-1]
        traj_smi = [m.smiles for m in traj]

        qed = self.qed_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        synth_score = self.synth_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        # stop optimizing qed/synth beyond thresholds
        clip_qed = LambdaZero.contrib.functional.satlins(qed, self.qed_cutoff[0], self.qed_cutoff[1])
        clip_synth = LambdaZero.contrib.functional.satlins(synth_score, self.synth_cutoff[0], self.synth_cutoff[1])
        proxy_dock, actor_info = self.dockProxy_actor([{"smiles":molecule.smiles, "mol_graph":molecule.graph,
                                                        "qed":qed, "synth_score":synth_score, "traj_smi":traj_smi,
                                                        "env_name": self.env_name}], [clip_qed * clip_synth])
        reward = float(proxy_dock[0]) * clip_qed * clip_synth

        info = {
            "proxy_dock": proxy_dock,
                "proxy_dock_mean": actor_info["mean"][0],
                "proxy_dock_var": actor_info["var"][0],
                "synth_score": synth_score, "qed_score":qed,
                "clip_qed": clip_qed, "clip_synth": clip_synth}
        return reward, info

    def __call__(self, traj, agent_stop, env_stop):
        return self.eval(traj)



class ProxyRewardSparse_v2(ProxyReward_v2):
    def __call__(self, traj, agent_stop, env_stop):
        if agent_stop or env_stop:
            reward, info = ProxyReward_v2.eval(self, traj)
        else:
            reward, info = 0.0, {}
        return reward, info



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
        qed = self.qed_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        synth_score = self.synth_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        # stop optimizing qed/synth beyond thresholds
        clip_qed = LambdaZero.contrib.functional.satlins(qed, self.qed_cutoff[0], self.qed_cutoff[1])
        clip_synth = LambdaZero.contrib.functional.satlins(synth_score, self.synth_cutoff[0], self.synth_cutoff[1])
        proxy_dock, actor_info = self.dockProxy_actor([{"smiles":molecule.smiles, "mol_graph":molecule.graph,
                                                        "qed":qed, "synth_score":synth_score,
                                                        "env_name": self.env_name}], [clip_qed * clip_synth])

        reward = float(proxy_dock[0]) * clip_qed * clip_synth

        info = {
            "molecule_num_blocks": len(molecule.jbond_atmidxs),
            "molecule_num_branches":len(molecule.stems),
            "molecule_num_atoms":molecule.slices[-1],
            "proxy_dock": proxy_dock,
                "proxy_dock_mean": actor_info["mean"][0],
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
            info["ended_on_env_stop"] = float(env_stop)
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