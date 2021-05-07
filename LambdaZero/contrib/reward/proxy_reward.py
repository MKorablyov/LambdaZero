import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config
import ray
from copy import deepcopy

from LambdaZero.inputs.inputs_op import _brutal_dock_proc
from LambdaZero.contrib.config_acquirer import LOG_CANDIDATE_QED_SCORE, LOG_CANDIDATE_SYNTH_SCORE
from LambdaZero.contrib.oracle.oracle import DockVina_smi_db

import LambdaZero.contrib.functional




class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff, synth_options, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
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
        clip_qed = LambdaZero.contrib.functional.satlins(qed, self.qed_cutoff[0], self.qed_cutoff[1])
        clip_synth = LambdaZero.contrib.functional.satlins(synth_score, self.synth_cutoff[0], self.synth_cutoff[1])
        proxy_dock, actor_info = self.dockProxy_actor([{"smiles": molecule.smiles,
                                                        "qed": qed,
                                                        "synth_score": synth_score,
                                                        "mol_graph": molecule.graph,
                                                        "blockidxs": molecule.blockidxs,
                                                        "slices": molecule.slices,
                                                        "jbonds": molecule.jbonds,
                                                        "stems": molecule.stems,
                                                        "env_name": self.env_name}],
                                                      [clip_qed * clip_synth])

        reward = float(proxy_dock[0]) * clip_qed * clip_synth

        # log feasible
        cand_dock_mean = actor_info["mean"][0] \
            if qed > self._log_cand_qed_th and synth_score > self._log_cand_synth_th else 0

        info = {
            "molecule_num_blocks": len(molecule.jbond_atmidxs),
            "molecule_num_branches":len(molecule.stems),
            "molecule_num_atoms": molecule.slices[-1],
            "proxy_cand_dock_mean": cand_dock_mean,
            "proxy_dock": proxy_dock,
            "proxy_dock_mean": actor_info["mean"][0],
            "proxy_dock_var": actor_info["var"][0],
            "synth_score": synth_score, "qed_score":qed,
            "clip_qed": clip_qed, "clip_synth": clip_synth,
            **actor_info  # TODO not sure how much it clogs
        }

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




