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

from LambdaZero.examples.hobby.proxy import Actor
from LambdaZero.examples.hobby.acquisition_function import UCB

class ProxyReward:
    def __init__(self, scoreProxy, actor_sync_freq, **kwargs):
        self.actor = Actor(scoreProxy, actor_sync_freq)

    def reset(self):
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        synth_score = 0.5
        qed = 0.9

        dock_score = self.actor([molecule], [qed * synth_score])[0]
        print("received dock_score from actor", dock_score)
        scores = {"dock_score":dock_score, "synth_score": synth_score, "qed":0.9}

        return synth_score * dock_score * qed, scores

