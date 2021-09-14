import pdb
import time

from collections import deque

from argparse import Namespace
from typing import List, Any
import numpy as np
import torch
from torch.multiprocessing import Process, Queue
import os.path as osp
from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended
import random
from LambdaZero.utils import get_external_dirs
from collections import Counter
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended


datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class DataFilterAddOnline(DataGenerator):
    def _add_mol_to_online(self, r: float, score: float, m: BlockMoleculeDataExtended, actually_add=False):
        if actually_add:
            self.online_mols.append((r, score, m))
            if len(self.online_mols) > self.max_online_mols:
                # TODO D no need for sort. just keep the latest ones
                self.online_mols = self.online_mols[int(self.max_online_mols * 0.2):]
