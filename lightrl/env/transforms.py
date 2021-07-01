from typing import List, Dict, Tuple
import numpy as np

from lightrl.env.oracle import InterogateOracle
from LambdaZero.contrib.functional import elu2, satlins


class TransformInfoOracle:
    def __init__(self, num_workers=10, send_updates_conn=None):
        self.oracle = InterogateOracle(num_workers=num_workers, send_updates_conn=send_updates_conn)

    def __call__(self, infos: List[dict]):
        _match = dict({x["mol"]["smiles"]: idx for idx, x in enumerate(infos)})
        self.oracle([x["mol"] for x in infos])
        ret_scores = self.oracle.get_req()
        for smi, dockscore in ret_scores.items():
            infos[_match[smi]]["dockscore"] = dockscore
        return infos


class TransformInfoDiscounted:
    def __init__(self,
                 score_key="dockscore", mean=-8.6, std=1.1, act_y=elu2,
                 qed_cutoff=(0.2, 0.5), synth_cutoff=(0, 4)):
        self._score_key = score_key
        self._mean = mean
        self._std = std
        self._act_y = act_y
        self._qed_cut = qed_cutoff
        self._synth_cut = synth_cutoff

    def __call__(self, infos: List[dict]):
        sk = self._score_key
        filter = [True if (sk in x and x[sk] is not None) else False for x in infos]
        idxs = np.where(filter)[0]
        scores = [infos[x][sk] for x in idxs]
        scores = [(self._mean - d) / self._std for d in scores]  # this normalizes and flips dockscore
        scores = np.array(self._act_y(scores))
        clip_qed = np.array([
            satlins(q, self._qed_cut[0], self._qed_cut[1]) for q in [infos[x]["qed"] for x in idxs]
        ])

        clip_synth = np.array([
            satlins(q, self._synth_cut[0], self._synth_cut[1]) for q in [infos[x]["synth"] for x in idxs]
        ])

        scores = scores * clip_synth * clip_qed
        for idx, info in enumerate(infos):
            info["dscore"] = None

        for idx, score in zip(idxs, scores):
            infos[idx]["dscore"] = score

        return infos


class TransformCompose:
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, infos: List[dict]):
        for x in self._transforms:
            infos = x(infos)
        return infos
