import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


# parser = argparse.ArgumentParser()

# parser.add_argument("--save_path", default='results/test_bits_q_1.pkl.gz', type=str)
# parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
# parser.add_argument("--learning_method", default='is_xent_q', type=str)
# parser.add_argument("--opt", default='adam', type=str)
# parser.add_argument("--adam_beta1", default=0.9, type=float)
# parser.add_argument("--adam_beta2", default=0.999, type=float)
# parser.add_argument("--momentum", default=0.9, type=float)
# parser.add_argument("--bootstrap_tau", default=0.1, type=float)
# parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
# parser.add_argument("--horizon", default=8, type=int)
# parser.add_argument("--n_hid", default=256, type=int)
# parser.add_argument("--n_layers", default=2, type=int)
# parser.add_argument("--n_train_steps", default=10000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
# parser.add_argument("--uniform_sample_prob", default=0.05, type=float)
# parser.add_argument("--do_is_queue", action='store_true')
# parser.add_argument("--queue_thresh", default=10, type=float)
# parser.add_argument("--device", default='cpu', type=str)
# parser.add_argument("--progress", action='store_true')

# global count
# count = 0

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
        return 0

    def get_scoring_reward(self, obs):
        # print("inside scoring reward")
        # print("state = ", obs)
        # print("obs shape = ", obs.shape)
        obs = obs.copy().squeeze(1)
        x = self.func(obs.copy())
        # print("x = ", x)
        # print("x shape = ", x.shape)
        reward = np.sum(x)
        return (reward/self.particle_size)

    def step(self, a):
        self.int_curr_time_step = self.int_next_time_step
        done = False
        # print("bitmap shape = ", self.bitmap.shape)
        # print("timestep shape = ", self.time_step.shape)
        step_size = np.expand_dims(self.bitmap, axis=1) * self.time_step
        step_size = step_size.squeeze(1)
        step_size = np.sum(step_size, axis=0)
        # step_size = self.bitmap.expand_dims(1) * self.time_step
        # print("step_size = ", step_size)
        # print("step size shape = ", step_size.shape)
        # print("action shape = ", a.shape)
        action_step = a * step_size
        # print("action step = ", action_step)
        # print("before: self.obs shape = ", self.obs.shape)
        self.obs = self.obs + np.expand_dims(action_step, axis=1)
        # print("after: self.obs shape = ", self.obs.shape)
        # self.obs = self.obs.squeeze(1)
        # print("after: self.obs shape = ", self.obs.shape)
        # print("after: self.obs = ", self.obs)
        diversity_reward = self.get_diversity_reward(self.obs)
        scoring_reward = self.get_scoring_reward(self.obs)
        reward = diversity_reward + scoring_reward
        # print("reward = ", reward)
        # int_time_step = np.argmax(self.time_step, axis=0)
        # print("int_time_Step = ", int_time_step)
        # self.time_step[int_time_step, 0] = 0
        self.time_step = np.zeros((self.horizon, 1))
        self.time_step[self.int_curr_time_step, 0] = 1.0
        self.state = np.concatenate((self.obs, self.time_step), 0)

        if self.int_curr_time_step == self.horizon-1:
            done = True
            return self.state, reward, done, None

        else:
            self.int_next_time_step = self.int_next_time_step + 1
            return self.state, reward, done, None



# def main(args):
#     env = DiversityEnv(args.horizon)
#     s = env.reset()
#     dev = torch.device(args.device)
#     env.step(np.array([-1, 1, 1, -1]))

# if __name__ == "__main__":
#     args = parser.parse_args()
#     torch.set_num_threads(1)
#     main(args)
