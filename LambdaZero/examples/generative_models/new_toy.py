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


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_bits_q_1.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='is_xent_q', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.05, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')


class BinaryTreeEnv:

    def __init__(self, horizon, ndim=1, xrange=[-1, 1], func=None):
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

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        return s

    def s2x(self, s):
        return (self.obs(s).reshape((self.horizon, 2)) * self.bitmap_mul).sum()

    def reset(self):
        self._state = []
        self._step = 0
        return self.obs()

    def step(self, a, s=None):
        _s = s
        s = self._state if s is None else s
        s.append(a)

        done = len(s) >= self.horizon
        if _s is None:
            self._state = s
        return self.obs(), 0 if not done else self.func(self.s2x(s)), done, s


def main(args):
    env = BinaryTreeEnv(args.horizon)
    s = env.reset()
    dev = torch.device(args.device)
    env.step(0)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
