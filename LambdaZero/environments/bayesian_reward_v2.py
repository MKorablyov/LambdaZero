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


@ray.remote(num_gpus=0.25, num_cpus=2)
class ScoreProxy:
    "combine scores from many models with uncertainty"
    def __init__(self, update_freq):
        self.update_freq = update_freq
        self.proposed_molecules = []
        self.proposed_scores = []
        #
        self.dockScore = DockScoreUCB()

    def propose_molecule(self, molecule, scores):
        self.proposed_molecules.append(molecule)
        self.proposed_scores.append(scores)

        if len(self.proposed_molecules) == self.update_freq:
            self.acquire_and_update()
        print(len(self.proposed_molecules))
        return None

    def acquire_and_update(self):
        print("updating proxy", len(self.proposed_molecules))
        self.dockScore.acquire_and_update(self.proposed_molecules, self.proposed_scores)
        self.proposed_molecules, proposed_scores = [], []

    def get_actor(self):
        # molecule
        # score = self.DocScoreUCB(molecule) * self.QED(molecule) * self.Synth(molecule)
        pass


class UCB:
    def __init__(self):
        # make model
        pass

    def acquire_batch(self, molecules, discounts):
        pass

    def update_with_seen(self):
        pass


class DockScoreUCB(UCB):
    name = "dock_score"
    # todo: log acquisitions

    def __call__(self, molecule):
        # call UCB with a single example
        pass

    def acquire(self, molecules):
        # do docking here
        pass

    def acquire_and_update(self, molecules, scores):
        # take_highest_k(molecules, scores)
        # dockscores = self.acquire(highest_scoring_molecules)
        print("dock score UCB to acquire molecules", len(molecules))
        # self.update_with_seen(molecules, dockscores)
        return None


class ProxyActor:
    def __init__(self):
        self.sync_weights()

    def compute_reward(self):
        pass

    def sync_weights(self):
        pass




# todo: I want to be able to:
#  a) share bottom weights including with agent
#  b) have any model of uncertainty such as BRR/evidential/etc/docking and any acquisition function
#  c) I want to be able to run the same acquisition function on a static dataset
#  d*) I want to run BRR in online (just choose a small update interval should be fine)


# Reward
# __init__(CombinedScoreInstance)
#   self.reward_actor = CombinedScoreInstance.get_actor()
#
# __call__(molecule)
#   combined_score_instance.add_molecule()
#   return self.reward_actor(molecule)


# RewardProxy@remote
# __init__(DockScoreUCB, SynthScore)
# add_molecule()
#   if 1000 molecules:
#       idx = self.acquire_and_update()
#
# def acquire_and_update():
#   logs = self.DockScore.acquire_and_update(self.molecules, self.scores, self.discounts)
#   return logs
#
# def get_actor(): # this will make a local actor to compute scores
#   return rewardActor


# DockScoreUCB(UCB):
#
# acquire():
#   do_docking
#
# acquire_and_update(molecules, scores, discounts)
#    molecules, aq_vals, discounts = self.acquire_batch()
#    self.dock(molecules)
#    self.update_with_seen(molecules, dockscores, discounts)


# UCB:
# __init__(model, dataset)
#
# update_with_seen(molecules, scores, discounts)
#   train network on data
#
# get_scores()
#
# acquire_batch(molecules, scores, discounts)
#

# ProxyScoreActor
# __init__(get_weights_func):
# __call__(molecule):
#   if_update_weights:
    # welf.weights = self.get_weights_func.remote.get_weights()
    # return score