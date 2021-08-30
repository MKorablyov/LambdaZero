import argparse
from copy import copy, deepcopy
from collections import defaultdict
from datetime import timedelta
import gc
import gzip
import os
import os.path as osp
import pickle
import psutil
import pdb
import subprocess
import sys
import threading
import time
import traceback
import warnings
warnings.filterwarnings('ignore')
import lmdb
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn
from torch_scatter import scatter_max

import LambdaZero.models
from LambdaZero import chem
from LambdaZero.chem import chem_op
from LambdaZero.environments.molMDP import BlockMoleculeData, MolMDP
#from LambdaZero.environments.reward import PredDockReward
#from LambdaZero.environments.reward import PredDockReward_v3 as PredReward
from LambdaZero.environments.block_mol_v3 import DEFAULT_CONFIG as env_v3_cfg, BlockMolEnv_v3
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config
from LambdaZero.utils import get_external_dirs

from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended

import model_atom, model_block, model_fingerprint

from main_flow import Dataset as GFNDataset, Proxy, make_model
from train_proxy import Dataset as PDataset

tmp_dir = "/tmp/lambdazero"

os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.999, type=float)
parser.add_argument("--opt_epsilon", default=1e-8, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--nembmol", default=56, help="#mol embedding", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--num_iterations", default=20000, type=int)
parser.add_argument("--num_conv_steps", default=6, type=int)
parser.add_argument("--log_reg_c", default=1e-2, type=float)
parser.add_argument("--reward_exp", default=4, type=float)
parser.add_argument("--reward_norm", default=8, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--R_min", default=1e-8, type=float)
parser.add_argument("--leaf_coef", default=1, type=float)
parser.add_argument("--clip_grad", default=0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--replay_mode", default='dataset', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--random_action_prob", default=0, type=float)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v6')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--save_path", default='results/')
parser.add_argument("--proxy_path", default='data/proxy/')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')
parser.add_argument("--floatX", default='float64')
parser.add_argument("--include_nblocks", default=True)
parser.add_argument("--balanced_loss", default=True)
# If True this basically implements Buesing et al's TreeSample Q,
# samples uniformly from it though, no MTCS involved
parser.add_argument("--do_wrong_thing", default=False)


class DatasetVAE(PDataset):
    def _get(self, i, dset):
        return [(dset[i], [(t[0], t[1]) for t in dset[i].traj])]

    def sample2batch(self, mb):
        s, tbits = mb
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        parents = self.mdp.mols2batch([self.mdp.mol2repr(c[0]) for i in tbits for c in i])
        actions = torch.tensor([c[1] for i in tbits for c in i])
        state_index = torch.tensor(sum([[i] * len(p) for i, p in enumerate(tbits)],[]))
        return s, parents, actions, state_index
        

    

def train_model_with_proxy(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = True
    device = torch.device('cpu')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    if do_save:
        exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
        os.makedirs(exp_dir, exist_ok=True)


    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)

    def save_stuff():
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))


    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2),
                           eps=args.opt_epsilon)
    #opt = torch.optim.SGD(model.parameters(), args.learning_rate)

    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything
    
    nworks = 0

    for ex in tqdm(dataset.train_mols):
        def _f(m):
            if not len(m.blocks):
                return True, []
            parents = dataset.mdp.parents(m)
            for p in parents:
                c = dataset.mdp.add_block_to(p[0], *p[1])
                if tuple(c.blockidxs) == tuple(m.blockidxs):
                    works, traj = _f(p[0])
                    if works: 
                        return True, traj + [p]
            return False, []

        works, traj = _f(ex)
        if works:
            nworks += 1
            ex.traj = traj + [(ex, (-1, 0))]
        else:
            ex.traj = None

    print(nworks)
                
            
                       
    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    loginf = 1000 # to prevent nans
    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])
    balanced_loss = args.balanced_loss
    do_nblocks_reg = False
    max_blocks = args.max_blocks
    leaf_coef = args.leaf_coef
    nemb = args.nemb
    nembmol = args.nembmol
    logsoftmax = nn.LogSoftmax(1)

    for i in tqdm(range(num_steps), disable=not args.progress):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
                    return
            s, r = r
        else:
            s, parents, actions, state_index = dataset.sample2batch(dataset.sample(mbsize))
        # asd
        _, s_enc = model(s, vec_data=model.default_emb[torch.zeros(s.num_graphs).long()], do_stems=False)
        s_enc_mean = s_enc[:, 1:nembmol+1]
        s_enc_logvar = s_enc[:, nembmol+1:]
        #h = torch.randn_like(s_enc_logvar) * torch.exp(s_enc_logvar * 0.5) + s_enc_mean
        h = torch.tanh(s_enc_mean)
        
        stem_logits, mol_logits = model(parents, vec_data=h[state_index], do_stems=True)
        neglogp = model.action_negloglikelihood(parents, actions, None, stem_logits, mol_logits)
        #loss = neglogp.mean() - 0.5 * (1 + s_enc_logvar - s_enc_logvar.exp() - s_enc_mean.pow(2)).mean()
        loss = neglogp.mean() + s_enc_mean.pow(2).mean() * 1e-3
        loss.backward()
        opt.step()
        opt.zero_grad()

        def compute_error(actions, stem_logits, mol_logits, parents):
            stem_max, stem_argmax = stem_logits.max(1)
            which_max, which_argmax = scatter_max(stem_max, parents.stems_batch)
            stem_choice = which_argmax - torch.tensor(parents.__slices__['stems'][:-1])
            block_choice = stem_argmax[which_argmax]
            predicted_actions = torch.stack([block_choice, stem_choice]).T
            stop_choice = (mol_logits[:, 0] > which_max)
            predicted_actions[stop_choice] = torch.tensor([-1, 0])
            correct = (predicted_actions != actions).prod(1).float()
            error = 1 - correct.mean()
            return error




        if not i % 50:
            error = compute_error(actions, stem_logits, mol_logits, parents)
            error_no_noise = compute_error(
                actions, 
                *model(parents, vec_data=s_enc_mean[state_index], do_stems=True),
                parents)
            print(i, loss.item(), error, error_no_noise)
            print(h.mean().item(), h.pow(2).mean().sqrt().item(), h.var().item())
        if not i % 500 and do_save:
            save_stuff()

    save_stuff()
                       
    



def main(args):
    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    device = torch.device('cpu')

    if args.floatX == 'float32':
        args.floatX = torch.float
    else:
        args.floatX = torch.double
    dataset = DatasetVAE(args, bpath, device, floatX=args.floatX)
    dataset.load_h5("docked_mols.h5", args, num_examples=100000)
    print(args)


    mdp = dataset.mdp
    #synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)
    #past_ep = pickle.load(gzip.open('replays/latest_oct_22_5.pkl.gz', 'rb'))
    #replay.load_initial_episodes(past_ep)

    model = make_model(args, mdp, out_per_mol=args.nembmol*2+1, nvec=args.nembmol)
    model.to(args.floatX)
    model.to(device)

    proxy = Proxy(args, bpath, device)

    train_model_with_proxy(args, model, proxy, dataset, do_save=True)
    print('Done.')


try:
    from arrays import*
except:
    print("no arrays")

good_config = {
    'replay_mode': 'online',
    'sample_prob': 1,
    'mbsize': 4,
    'max_blocks': 8,
    'min_blocks': 2,
    # This repr actually is pretty stable
    'repr_type': 'block_graph',
    'model_version': 'v4',
    'nemb': 256,
    # at 30k iterations the models usually have "converged" in the
    # sense that the reward distribution doesn't get better, but the
    # generated molecules keep being unique, so making this higher
    # should simply provide more high-reward states.
    'num_iterations': 30000,

    'R_min': 0.1,
    'log_reg_c': (0.1/8)**4,
    # This is to make reward roughly between 0 and 1 (proxy outputs
    # between ~0 and 10, but very few are above 8). Maybe you will
    # need to adjust for uncertainty?
    'reward_norm': 8,
    # you can play with this, higher is more risky but will give
    # higher rewards on average if it succeeds.
    'reward_exp': 10,
    'learning_rate': 5e-4,
    'num_conv_steps': 10,
    # I only tried this and 0, I'm not sure there is much difference
    # but in priciple exploration is good
    'random_action_prob': 0.05,
    'opt_beta2': 0.999,
    'leaf_coef': 10, # I only tried 1 and 10, 10 works pretty well
    'include_nblocks': False,
}

if __name__ == '__main__':
  args = parser.parse_args()
  _stop = [lambda:None]
  if 0:
    all_hps = eval(args.array)(args)
    #for run in range(66,69):
    for run in range(69, 72):
    #for run in range(len(all_hps)):
      args.run = run
      hps = all_hps[run]
      for k,v in hps.items():
        setattr(args, k, v)
      exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
      #if os.path.exists(exp_dir):
      #  continue
      print(hps)
      main(args)
  elif args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
    else:
      hps = all_hps[args.run]
      print(hps)
      for k,v in hps.items():
        setattr(args, k, v)
    try:
        main(args)
    except KeyboardInterrupt as e:
        print("stopping for", e)
        _stop[0]()
        raise e
    except Exception as e:
        print("exception", e)
        _stop[0]()
        raise e
  else:
      try:
          main(args)
      except KeyboardInterrupt as e:
          print("stopping for", e)
          _stop[0]()
          raise e
      except Exception as e:
          print("exception", e)
          _stop[0]()
          raise e
