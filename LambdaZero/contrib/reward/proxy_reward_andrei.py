import time
import numpy as np
from LambdaZero.contrib.proxy import Actor
from rdkit import Chem
from random import random
from LambdaZero.contrib.oracle import QEDOracle, SynthOracle
from LambdaZero.environments.block_mol_v3 import synth_config
import ray
from copy import deepcopy

from LambdaZero.contrib.config_acquirer import LOG_CANDIDATE_QED_SCORE, LOG_CANDIDATE_SYNTH_SCORE
from LambdaZero.contrib.oracle.oracle import DockVina_smi_db

from LambdaZero.contrib.reward.proxy_reward import ProxyReward


class DockingEstimatorThread(DockVina_smi_db):
    """ Docking estimator separate for each env. """
    def __init__(self, dockVina_config):
        DockVina_smi_db.__init__(self, **dockVina_config)

    def eval(self, smiles, **kwargs):
        try:
            mol_name, dockscore, coord = self.dock(smiles, **kwargs)
        except Exception as e:
            dockscore = None
        return dockscore


class InterogateOracle:
    """
        Interogate global thread for looking up predocked scores. (1 db loaded)
        If not found use local thread to calculate oracle (+update db)
    """

    def __init__(self, scoreProxy):
        self._scoreProxy = scoreProxy

        self._scoreProxy.set_oracle_fix_none.remote(False)
        self._scoreProxy.set_oracle_norm_data.remote(False)

        self._oracle_cfg = oracle_cfg = ray.get(self._scoreProxy.get_oracle_config.remote())
        cfg = deepcopy(oracle_cfg["dockVina_config"])
        cfg["query_predocked"] = False
        self._true_oracle = DockingEstimatorThread(cfg)
        self.mean = oracle_cfg["mean"]
        self.std = oracle_cfg["std"]
        self.act_y = oracle_cfg["act_y"]

    def __call__(self, x, d):
        _id = np.random.randint(10000)
        # print(f"GET({_id})", x)
        # Send only
        smi_data = [{"smiles": i["smiles"]} for i in x]
        res = ray.get(self._scoreProxy.get_oracle.remote(smi_data))

        calc_new = [False for _ in range(len(x))]

        for i, (smi, data, dockscore) in enumerate(zip(smi_data, x, res)):
            if dockscore is None:
                new_dockscore = self._true_oracle.eval(smi["smiles"], mol_data=data)
                if new_dockscore is None:
                    # Add to failed hits
                    print(f"Cannot get new reward ------->>>> "
                          f"({_id}): {smi['smiles']} {new_dockscore}_")

                self._scoreProxy.update_predocked.remote(smi["smiles"], new_dockscore)

                res[i] = new_dockscore
                calc_new[i] = True

        dockscores_ = []
        num_failures = 0
        for d in res:
            if d is None:
                dockscores_.append(self.mean)  # mean on failures
                num_failures += 1
            else:
                dockscores_.append(d)

        dockscores = [(self.mean-d) / self.std for d in dockscores_]  # this normalizes and flips dockscore

        dockscores = self.act_y(dockscores)

        res = [333 if x is None else x for x in res]
        return dockscores, dict({"mean": [0]*len(x), "var": [0]*len(x), "true_dockscore": res})


class ProxyRewardOradcle(ProxyReward):
    """ Use common db for predocked scores and local thread for new molecules"""
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff, synth_options, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.qed_oracle = QEDOracle(num_threads=1)
        self.synth_oracle = SynthOracle(synth_options, synth_config)
        self.dockProxy_actor = InterogateOracle(scoreProxy)

        # Candidate Molecule hard thresholds (Same in Proxy)
        # Should be fixed in order to be able to compare runs across different training configs
        self._log_cand_qed_th = LOG_CANDIDATE_QED_SCORE
        self._log_cand_synth_th = LOG_CANDIDATE_SYNTH_SCORE

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        if agent_stop or env_stop:
            reward, info = ProxyReward.eval(self, molecule)
            info["ended_on_env_stop"] = float(env_stop)
        else:
            reward, info = 0.0, {}
        return reward, info


class DummyOracle:
    """ 0 score oracle """
    def __init__(self, scoreProxy):
        self._scoreProxy = scoreProxy

    def __call__(self, x, d):
        dockscores = [0] * len(x)
        res = [0] * len(x)
        return dockscores, dict({"mean": [0]*len(x), "var": [0]*len(x), "true_dockscore": res})


class ProxyRewardNoDocking(ProxyReward):
    """ 0 for docking score """
    def __init__(self, scoreProxy, actor_sync_freq, qed_cutoff, synth_cutoff, synth_options, **kwargs):
        self.env_name = np.random.uniform()
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.qed_oracle = QEDOracle(num_threads=1)
        self.synth_oracle = SynthOracle(synth_options, synth_config)
        self.dockProxy_actor = DummyOracle(scoreProxy)

        # Candidate Molecule hard thresholds (Same in Proxy)
        # Should be fixed in order to be able to compare runs across different training configs
        self._log_cand_qed_th = LOG_CANDIDATE_QED_SCORE
        self._log_cand_synth_th = LOG_CANDIDATE_SYNTH_SCORE

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        if agent_stop or env_stop:
            reward, info = ProxyReward.eval(self, molecule)
            info["ended_on_env_stop"] = float(env_stop)
        else:
            reward, info = 0.0, {}
        return reward, info


class DummyReward:
    """ Dummy reward """
    def __init__(self, **kwargs):
        self.qed_oracle = QEDOracle(num_threads=1)

    def reset(self, previous_reward=0.0):
        self.previous_reward = 0.0
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        return float(random()), {"reward": 1.0, "discounted_reward": 1.0, "QED": 1.0, "discount": 1.0}