import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config
import ray
from LambdaZero.inputs.inputs_op import _brutal_dock_proc
from LambdaZero.contrib.config_acquirer import LOG_CANDIDATE_QED_SCORE, LOG_CANDIDATE_SYNTH_SCORE


def _satlins(x, cutoff0, cutoff1):
    "shifted saturated linearity activation function _/-"
    x = (x - cutoff0) / (cutoff1 - cutoff0)
    x = min(max(0.0, x), 1.0)  # relu to maxout at 1
    return x

class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff,
                 exp_dock, always_discount, synth_options, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.exp_dock = exp_dock
        self.always_discount = always_discount
        self.qed_oracle = QEDOracle(num_threads=1)
        self.synth_oracle = SynthOracle(synth_options, synth_config)
        self.dockProxy_actor = Actor(scoreProxy, actor_sync_freq)

        # Candidate Molecule hard thresholds (Same in Proxy)
        # Should be fixed in order to be able to compare runs across different training configs
        self._log_cand_qed_th = LOG_CANDIDATE_QED_SCORE
        self._log_cand_synth_th = LOG_CANDIDATE_SYNTH_SCORE

    def reset(self, previous_reward=0.0):
        return None

    def eval(self, molecule):
        qed = self.qed_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        synth_score = self.synth_oracle([{"smiles":molecule.smiles, "mol":molecule.mol}])[0]
        # stop optimizing qed/synth beyond thresholds
        clip_qed = _satlins(qed, self.qed_cutoff[0], self.qed_cutoff[1])
        clip_synth = _satlins(synth_score, self.synth_cutoff[0], self.synth_cutoff[1])

        proxy_dock, actor_info = self.dockProxy_actor([{"smiles":molecule.smiles,
                                                        "mol_graph":molecule.graph,
                                                        "env_name": self.env_name,
                                                        "qed": qed,
                                                        "synth_score": synth_score,}],
                                                      [clip_qed * clip_synth])
        proxy_dock = float(proxy_dock[0]) # actor works on multiple x by default

        if self.exp_dock:
            reward = (self.exp_dock ** proxy_dock) * clip_qed * clip_synth
        elif self.always_discount:
            reward = proxy_dock * clip_qed * clip_synth
        elif proxy_dock > 0: # reward should be rarely negative; when negative, discount won't be applied
            reward = proxy_dock * clip_qed * clip_synth
        else:
            reward = proxy_dock

        # log feasible
        cand_dock_mean = actor_info["mean"][0] \
            if qed > self._log_cand_qed_th and synth_score > self._log_cand_synth_th else 0

        info = {
            "molecule_num_blocks": len(molecule.jbond_atmidxs),
            "molecule_num_branches":len(molecule.stems),
            "molecule_num_atoms":molecule.slices[-1],
            "proxy_cand_dock_mean": cand_dock_mean,
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