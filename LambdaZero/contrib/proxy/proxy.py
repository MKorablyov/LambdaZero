import time
import ray
from copy import deepcopy
import numpy as np
from typing import List, Dict
import collections
from LambdaZero.contrib.config_acquirer import LOG_CANDIDATE_QED_SCORE, LOG_CANDIDATE_SYNTH_SCORE


class BufferMolecules:
    """
        Buffer of molecules - used for stats so far.
        # TODO could implement here diversity metric
        # TODO could be extended to keep all the proposed.
    """
    def __init__(self, prefix_name: str, buffer_size: int = 1000):
        # Dict used for hashing [smiles] = hits
        # Dictionary lookup much faster, but dicts use a lot of memory.*
        # Speed could be improved by hashing first by num_atoms if needed
        self._prefix_name = prefix_name

        self._buffer_hash = dict({})
        self._max_dock = -np.inf
        self._max_dock_smiles = -np.inf
        self._unique_cnt = 0
        self._buffer_mol = collections.deque(maxlen=buffer_size)
        self._buffer_dock = collections.deque(maxlen=buffer_size)

    def update(self, evaluated_x: List[Dict], oracle_dock: np.ndarray) -> Dict:

        # Get max
        new_max = 0
        if len(oracle_dock) > 0:
            imax = np.argmax(oracle_dock).item()
            if oracle_dock[imax] > self._max_dock:
                self._max_dock = new_max = oracle_dock[imax]
                self._max_dock_smiles = evaluated_x[imax]["smiles"]

            for x, dock_score in zip(evaluated_x, oracle_dock):
                smiles = x["smiles"]
                self._buffer_mol.append(smiles)
                self._buffer_dock.append(dock_score)

                if smiles not in self._buffer_hash:
                    self._buffer_hash[smiles] = 1
                    self._unique_cnt += 1
                else:
                    self._buffer_hash[smiles] += 1

        m = self._prefix_name
        info = dict({f"proxy/{m}_new_max": new_max,
                     f"proxy/{m}_unique_count": self._unique_cnt,
                     f"proxy/{m}_buffer_uniq": len(set(self._buffer_mol)) / len(self._buffer_mol),
                     f"proxy/{m}_dock_oracle_mean": np.mean(self._buffer_dock),
                     f"proxy/{m}_dock_oracle_max": np.max(self._buffer_dock),
                     })

        return info


class Proxy:
    def __init__(self, update_freq, proposed_x, proposed_d, proposed_acq, logger):
        self.update_freq = update_freq
        self.proposed_x, self.proposed_d, self.proposed_acq = proposed_x, proposed_d, proposed_acq
        self.logger = logger
        self._model_id = 0

        self._log_cand_qed_th = LOG_CANDIDATE_QED_SCORE
        self._log_cand_synth_th = LOG_CANDIDATE_SYNTH_SCORE
        self._cand_buffer = BufferMolecules("cand")

    def propose_x(self,x, d, acq):
        """
        :param x: points to evaluate
        :param aq: current estimates of the acquisition value at x
        :param d: discount factors defaults to 1. This would be used for the acquisition
        :return:
        """
        self.proposed_x.extend(x)
        self.proposed_d.extend(d)
        self.proposed_acq.extend(acq)
        # update model and acquisition function if needed
        if len(self.proposed_x) >= self.update_freq:
            # todo: a better solution would be to
            #  keep and re-compute acqusition values of proposed while proxy-model was retraining
            #proposed_x, proposed_d, proposed_acq = self.proposed_x, self.proposed_d, self.proposed_acq
            #self.proposed_x, self.proposed_d, self.proposed_acq = [], [], []
            self.acquire_and_update(self.proposed_x, self.proposed_d, self.proposed_acq)
            self.proposed_x, self.proposed_d, self.proposed_acq = [], [], []
            self._model_id += 1
        return None

    def acquire_and_update(self, proposed_x, proposed_d, proposed_acq):
        raise NotImplementedError

    def acquire(self, x):
        raise NotImplementedError

    def get_model(self):
        raise NotImplementedError

    def get_model_id(self):
        return self._model_id

    def post_acquire_and_update(self, evaluated_x: List[Dict], oracle_dock: List[int]) -> Dict:
        """ Log candidate (th by qed/synth) molecules oracle docking stats. Buffer oracle """

        qed = np.array([x["qed"] for x in evaluated_x])
        synth = np.array([x["synth_score"] for x in evaluated_x])
        oracle_dock = np.array(oracle_dock)

        selection = (qed > self._log_cand_qed_th) & (synth > self._log_cand_synth_th)
        candidate_dock = oracle_dock[selection]

        # Update candidate buffer
        cand_mol = [x for ix, x in enumerate(evaluated_x) if selection[ix]]
        # Show stats on fix length of candidates history
        buff_info = self._cand_buffer.update(cand_mol, candidate_dock)

        buff_info["proxy/cand_fraction"] = len(candidate_dock) / len(oracle_dock)

        return buff_info


class Actor():
    def __init__(self, scoreProxy, actor_sync_freq):
        self.scoreProxy = scoreProxy
        self.sync_freq = actor_sync_freq
        # initialize
        self.num_calls = 0
        self.acquisition_func = ray.get(scoreProxy.get_acquisition_func.remote())
        self.names = []
        self._acq_update_id = 0
        self._model_id = ray.get(self.scoreProxy.get_model_id.remote())

    def __call__(self, x, d):
        # compute acquisition value
        self.num_calls += 1
        acq, info = self.acquisition_func.acquisition_value(x)

        # send molecule to the remote proxy
        self.scoreProxy.propose_x.remote(deepcopy(x), deepcopy(d), deepcopy(acq))

        # sync weights with proxy if needed
        if self.num_calls % self.sync_freq==1:
            # todo - need to figure out to do non-blocking calls here
            self.acquisition_func = ray.get(self.scoreProxy.get_acquisition_func.remote())
            self._model_id = ray.get(self.scoreProxy.get_model_id.remote())

            self._acq_update_id += 1

        info["acq_update_id"] = self._acq_update_id
        info["model_id"] = self._model_id

        return acq, info