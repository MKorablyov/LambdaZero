import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
import sklearn
import sklearn.metrics

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class DiversityEnv(object):

    def __init__(self, horizon=8, ndim=1, xrange=[-1, 1], func=None):

        self.horizon = horizon
        self.center = sum(xrange)/2
        self.width = xrange[1] - xrange[0]
        self.func = (
            (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)) + 0.01)
            if func is None else func)

        self.bitmap = np.float32([1/2**(i+1) for i in range(horizon)])
        self.bitmap_mul = np.zeros((horizon, 2))
        self.bitmap_mul[:, 0] = self.bitmap
        self.bitmap_mul[:, 1] = -self.bitmap

        self.particle_size = 4
        self.k = 1
        self.action_database = []

        def PushTheString(arr):
            self.action_database.append(arr.copy())

        def generateAllBinaryStrings(n, arr, i):
            if i == n:
                PushTheString(arr)
                return
            arr[i] = -1.0
            generateAllBinaryStrings(n, arr, i + 1)
            arr[i] = 1.0
            generateAllBinaryStrings(n, arr, i + 1)

        arr = [None] * self.particle_size
        generateAllBinaryStrings(self.particle_size, arr, 0)
        self.action_database = np.array(self.action_database)

        self.obs = np.zeros((self.particle_size, 1))
        self.time_step = np.zeros((self.horizon, 1))
        self.state = np.concatenate((self.obs, self.time_step), 0)
        self.int_curr_time_step = 0
        self.int_next_time_step = 0

    def s2x(self, s):
        return (self.obs(s).reshape((self.horizon, 2)) * self.bitmap_mul).sum()

    def reset(self):
        self.obs = np.zeros((self.particle_size, 1))
        self.time_step = np.zeros((self.horizon, 1))
        self.time_step[0,0] = 1.0
        self.state = np.concatenate((self.obs, self.time_step), 0)
        self.int_curr_time_step = 0
        self.int_next_time_step = 0
        return self.state

    def get_diversity_reward(self, state):
        diversity_mat = sklearn.metrics.pairwise_distances(state.copy())
        reward = np.sum(np.concatenate(diversity_mat))
        return reward

    def get_scoring_reward(self, obs):
        obs = obs.copy().squeeze(1)
        x = self.func(obs.copy())
        reward = np.sum(x)
        return (reward/self.particle_size)

    def step(self, a):
        self.int_curr_time_step = self.int_next_time_step
        done = False
        step_size = np.expand_dims(self.bitmap, axis=1) * self.time_step
        step_size = step_size.squeeze(1)
        step_size = np.sum(step_size, axis=0)
        action_step = a * step_size
        self.obs = self.obs + np.expand_dims(action_step, axis=1)
        diversity_reward = self.get_diversity_reward(self.obs)
        scoring_reward = self.get_scoring_reward(self.obs)
        reward = diversity_reward + scoring_reward
        self.time_step = np.zeros((self.horizon, 1))
        self.time_step[self.int_curr_time_step, 0] = 1.0
        self.state = np.concatenate((self.obs, self.time_step), 0)

        if self.int_curr_time_step == self.horizon-1:
            done = True
            return self.state, reward, done, None

        else:
            self.int_next_time_step = self.int_next_time_step + 1
            return self.state, reward, done, None
