import pickle, gzip,time,  os.path as osp
import torch
import pandas as pd
import numpy as np
import ray

from torch_geometric.data import Batch, DataLoader
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import QED

import LambdaZero.utils
import LambdaZero.models
import LambdaZero.chem

class ProxyReward:
    def __init__(self, scoreProxy, **kwargs):
        self.score_proxy = scoreProxy

    def reset(self):
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        reward = 1.0
        scores = {"dock_score": 1.0, "synth_score": 1.0}
        self.score_proxy.propose_molecule.remote(molecule, scores)
        return reward, scores

