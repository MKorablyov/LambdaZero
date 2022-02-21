import cv2
from argparse import Namespace
import numpy as np
import torch
import random
from typing import List, Any, Tuple

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended

from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class DebugFlow(DataGenerator):

    @torch.no_grad()
    def _get_batch_reward(self, mmms: List[BlockMoleculeDataExtended]) -> Tuple[float, dict]:
        offset_r_blocks = self.args.offset_r_blocks
        res_scores, infos = self.proxy_reward(mmms)
        res_scores = self.r2r(dockscore=np.array(res_scores, dtype=np.float64))
        for ix, mmm in enumerate(mmms):
            if len(mmm.blockidxs) == offset_r_blocks:
                res_scores[ix] += 1

        return res_scores, infos
