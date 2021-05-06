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

from toy_1d_seq import BinaryTreeEnv, make_mlp, funcs
from toy_end2end import train_generative_model

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_bits_ndim_7.pkl.gz', type=str)
parser.add_argument("--reward_func", default='cos_sin_N', type=str)
parser.add_argument("--learning_rate", default=2e-4, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='td', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--mbsize", default=128, help="Minibatch size", type=int)
parser.add_argument("--horizon", default=5, type=int)
parser.add_argument("--num_dims", default=2, type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.001, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

def main(args):
    env = BinaryTreeEnv(args.horizon, func=funcs[args.reward_func], ndim=args.num_dims)
    policy, train_loss, val_loss, empirical_losses, all_visited = (
        train_generative_model(args, funcs[args.reward_func], None, None,
                               compute_empirical=True))
    pickle.dump((train_loss, empirical_losses, all_visited, args), gzip.open(args.save_path, 'wb'))

if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parser.parse_args()

    dev = torch.device(args.device)
    tf = lambda x: torch.FloatTensor(x).to(dev)
    tfc = lambda x: torch.FloatTensor(x)
    tl = lambda x: torch.LongTensor(x).to(dev)

    # Inject globals back into main... TODO: fixme :D
    import toy_end2end
    toy_end2end.dev = dev
    toy_end2end.tf = tf
    toy_end2end.tfc = tfc
    toy_end2end.tl = tl
    toy_end2end.LOGINF = torch.tensor(1000).to(dev)

    import toy_1d_seq
    import importlib
    importlib.reload(toy_end2end)
    importlib.reload(toy_1d_seq)

    main(args)
