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


class UCB:
    # todo: acquisition function - maybe create classes
    #  AcqusitionFunction; ModelWithUncertainty
    def __init__(self):
        # make model
        # seen_x, seen_y, val_x, val_y = None
        pass

    def update_with_seen(self, x, y):
        # self.seen_x +=x
        # self.model_with_uncertainty.fit(x,y, self.val_x, self.val_y)
        pass

    def acqusition_values(self, x):
        # mean, var = self.model.get_mean_and_variance(molecules)
        # return mean + kappa * var
        pass

    def acquire_batch(self, x, discounts, aq_values=None):
        # if aq_values = self.compute_acquisition_values(x)
        # aq_values_ = aq_values[top_k]
        # return idx
        pass


@ray.remote(num_gpus=0.25, num_cpus=2)
class ScoreProxy(UCB):
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



# class UCBTrainer(UCB, tune.trainable):
    # _init():
    #   seen_x, seen_y, val_x, val_y, unseen_x, unseen_y = .....
    # def train()
    #   idx  = self.acquire_batch(unseen_x)
    #   x_, y_ = unseen[idx], unseen[idx]
    #   self.update_with_seen(x_, y_)

class DockScoreUCB(UCB):
    #def __call__(self, molecule):
    #    # graph = G(molecule)
    #    # return self.acquisition_values(molecule)[0]

    def acquire(self, molecules):
        # do docking here
        # return molecules, dockscores
        pass

    def acquire_and_update(self, molecules, aq_scores, discounts):
        # idx = self.acquire_batch(molecules, aq_scores, discounts)
        # molecules[idx], discounts[idx]
        # dockscores = self.acquire(molecules)
        # self.update_with_seen(molecules, dockscores)
        print("dock score UCB to acquire molecules", len(molecules))
        return None

    def get_weights(self):
        pass


# todo: I want to be able to update every weight set independently with calls
# todo: maybe original score could become ray.remote() object

# DockScoreUCBActor(get_weights_func)
# get_weights()
# __call__(molecule)
#   if self.get_weights()
#   return self.weights * molecule



# class DockScoreUCBActor():
#     def __init__(self):
#         pass
#
#     def __call__(self, molecule):
#         if self.num_calls % self.weight_update_freq:
#             self.weights = self.get_weights_func()
#         return self.weights * molecule






class ProxyActor:
    def __init__(self, sync_freq, get_weights):
        self.sync_freq = sync_freq
        self.weights = get_weights.remote()

    def __call__(self, molecule):
        pass

    def _sync_weights(self):
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