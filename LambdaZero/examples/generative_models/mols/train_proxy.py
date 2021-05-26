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


'''
we could compare:
1- docking data as dataset
2- proxy to docking as R + sample p_theta
3- proxy to docking as R when x not in dataset (so iid? ood?) + sample p_theta
4- combine 1 & 3 as a mixture (because 3 might rarely stumble upon x in dataset)
'''

import importlib
importlib.reload(model_atom)
#importlib.reload(chem_op)

# datasets_dir, programs_dir, summaries_dir = get_external_dirs()
# if 'SLURM_TMPDIR' in os.environ:
#     print("Syncing locally")
#     tmp_dir = os.environ['SLURM_TMPDIR'] + '/lztmp/'

#     os.system(f"rsync -az {programs_dir} {tmp_dir}")
#     os.system(f"rsync -az {datasets_dir} {tmp_dir}")
#     programs_dir = f"{tmp_dir}/Programs"
#     datasets_dir = f"{tmp_dir}/Datasets"
#     print("Done syncing")
# else:
tmp_dir = "/tmp/lambdazero"

os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=64, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--nemb", default=64, help="#hidden", type=int)
parser.add_argument("--num_iterations", default=200000, type=int)
parser.add_argument("--num_conv_steps", default=12, type=int)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='atom_graph')
parser.add_argument("--model_version", default='v2')
parser.add_argument("--run", default=1, help="run", type=int)
#parser.add_argument("--save_path", default='/miniscratch/bengioe/LambdaZero/imitation_learning/')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')
parser.add_argument("--dump_episodes", default='')



from main_flow import Dataset as _Dataset


class Dataset(_Dataset):

    def _get(self, i, dset):
        return [(dset[i], dset[i].reward)]

    def itertest(self, n):
        N = len(self.test_mols)
        for i in range(int(np.ceil(N/n))):
            samples = sum((self._get(j, self.test_mols) for j in range(i*n, min(N, (i+1)*n))), [])
            yield self.sample2batch(zip(*samples))

    def sample2batch(self, mb):
        s, r, *o = mb
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        r = torch.tensor(r, device=self._device).float()
        return (s, r, *o)

    def load_h5(self, path, args, test_ratio=0.1, num_examples=None):
        import json
        import pandas as pd
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        store = pd.HDFStore(path, 'r')
        df = store.select('df')
        # Pandas has problem with calculating some stuff on float16
        df.dockscore = df.dockscore.astype("float64")
        for cl_mame in columns[2:]:
            df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)
        # if num_examples is None:

        # num_examples = len(df)
        # idxs = range(len(df))
        # else:
        #    idxs = self.test_split_rng.choice(len(df), int((1-test_ratio) * num_examples), replace=False)
        # Sample which indices will be our test set
        # idxs = range(len(df))
        test_idxs = self.test_split_rng.choice(len(df), int(test_ratio * len(df)), replace=False)

        split_bool = np.zeros(len(df), dtype=np.bool)
        split_bool[test_idxs] = True
        print("slit test", sum(split_bool), len(split_bool), "num examples", num_examples)

        for i in tqdm(range(len(df)), disable=not args.progress):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], df.iloc[i, c - 1])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            if len(m.blocks) > self.max_blocks:
                continue
            # TODO: compute proper reward with QED & all
            m.reward = self.r2r(dockscore=m.dockscore)
            m.numblocks = len(m.blocks)
            if split_bool[i]:
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
                self.train_mols_map[df.iloc[i].name] = m
            if len(self.train_mols) >= num_examples:
                break
        store.close()

    def load_pkl(self, path, args, test_ratio=0.05, num_examples=None):
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        mols = pickle.load(gzip.open(path))
        if num_examples is None:
            num_examples = len(mols)
            idxs = range(len(mols))
        else:
            idxs = self.test_split_rng.choice(len(mols), int((1 - test_ratio) * num_examples), replace=False)
        test_idxs = self.test_split_rng.choice(len(mols), int(test_ratio * num_examples), replace=False)
        split_bool = np.zeros(len(mols), dtype=np.bool)
        split_bool[test_idxs] = True
        for i in tqdm(idxs, disable=not args.progress):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], mols[i][columns[c]])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            if len(m.blocks) > self.max_blocks:
                continue
            # TODO: compute proper reward with QED & all
            m.reward = self.r2r(dockscore=m.dockscore)#max(self.R_min,4-(min(0, m.dockscore)-self.target_norm[0])/self.target_norm[1])
            m.numblocks = len(m.blocks)
            if split_bool[i]:
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
                self.train_mols_map[m.smiles if len(m.blocks) else '[]'] = m


def main(args):
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    device = torch.device('cuda')

    dataset = Dataset(args, bpath, device, floatX=torch.float)
    #dataset.load_h5("dock_db_1618610362tp_2021_04_16_17h.h5")
    dataset.load_h5("dock_db_1619111711tp_2021_04_22_13h.h5", args)
    #dataset.load_pkl("small_mols.pkl.gz", args)
    #dataset.load_pkl("small_mols_2.05.pkl.gz", args)

    exp_dir = f'{args.save_path}/proxy_{args.array}_{args.run}/'
    os.makedirs(exp_dir, exist_ok=True)
    print(args)
    debug_no_threads = False


    mdp = dataset.mdp
    #synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)
    #past_ep = pickle.load(gzip.open('replays/latest_oct_22_5.pkl.gz', 'rb'))
    #replay.load_initial_episodes(past_ep)
    print(len(dataset.train_mols), 'train mols')
    print(len(dataset.test_mols), 'test mols')

    stop_event = threading.Event()

    if args.repr_type == 'block_graph':
        mdp.post_init(device, args.repr_type)
        model = model_block.GraphAgent(nemb=args.nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=1,
                                       num_conv_steps=args.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version=args.model_version)
        model.to(device)
    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=args.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=args.num_conv_steps,
                                     version=args.model_version)
        model.to(device)
    elif args.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
        model.to(device)


    best_model = model
    best_test_loss = 1000

    opt = torch.optim.Adam(model.parameters(), args.learning_rate, #weight_decay=1e-4,
                           betas=(args.opt_beta, 0.999))

    tf = lambda x: torch.tensor(x, device=device).float()
    tint = lambda x: torch.tensor(x, device=device).long()

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    last_losses = []

    def stop_everything():
        stop_event.set()
        print('joining')
        dataset.stop_samplers_and_join()

    def save_stuff():
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump([i.data.cpu().numpy() for i in best_model.parameters()],
                    gzip.open(f'{exp_dir}/best_params.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

    train_losses = []
    test_losses = []
    test_infos = []
    time_start = time.time()
    time_last_check = time.time()
    nbatches_per_up = 1
    nbatches_done = 0
    batch_inc = 1000000000#5000
    next_batch_inc = batch_inc

    max_early_stop_tolerance = 5
    early_stop_tol = max_early_stop_tolerance
    loginf = 1000 # to prevent nans

    for i in range(args.num_iterations+1):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_event.set()
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
            s, r = r
        else:
            p, pb, a, r, s, d = dataset.sample2batch(dataset.sample(mbsize))

        # state outputs
        stem_out_s, mol_out_s = model(s, None, do_stems=False)
        loss = (mol_out_s[:, 0] - r).pow(2).mean()
        loss.backward()
        last_losses.append((loss.item(),))
        train_losses.append((loss.item(),))
        opt.step()
        opt.zero_grad()
        model.training_steps = i + 1
        #for _a,b in zip(model.parameters(), model_target.parameters()):
        #  b.data.mul_(1-tau).add_(tau*_a)

        if not i % 1000:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses)
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

            if i % 5000:
                continue

            if 0:
              save_stuff()
              continue

            t0 = time.time()
            total_test_loss = 0
            total_test_n = 0
            all_nlls = []
            all_ts = []
            for s, r in dataset.itertest(max(mbsize, 128)):
                with torch.no_grad():
                    stem_o, mol_o = model(s, None, do_stems=False)
                    loss = (mol_o[:, 0] - r).pow(2)
                    total_test_loss += loss.sum().item()
                    total_test_n += loss.shape[0]
            test_loss = total_test_loss / total_test_n
            if test_loss < best_test_loss:
              best_test_loss = test_loss
              best_model = deepcopy(model)
              best_model.to('cpu')
              early_stop_tol = max_early_stop_tolerance
            else:
              early_stop_tol -= 1
            print('test loss:', test_loss)
            print('test took:', time.time() - t0)
            test_losses.append(test_loss)
            save_stuff()
            if early_stop_tol <= 0 and False:
              print("Early stopping")
              break


    stop_everything()
    save_stuff()
    print('Done.')



def array_feb_23(args):
  all_hps = ([
    {'mbsize': 64,
     'learning_rate': 1e-4,
     'num_iterations': 200_000,
     'save_path': './results/',
     'nemb': 64,
     'repr_type': 'block_graph',
     'model_version': 'v3',
     'num_bins': 8,
     'top_bin': top_bin,
     'include_qed_data': qed,
     'mol_data': mol_data,
     }
    for _run in [0,1]
    for qed in [True, False]
    for top_bin in range(0,8)
    for mol_data in ['10k', '20k']
  ])
  return all_hps

if __name__ == '__main__':
  args = parser.parse_args()
  if 0:
    all_hps = eval(args.array)(args)
    #for run in range(0,8):
    for run in range(len(all_hps)):
      args.run = run
      hps = all_hps[run]
      for k,v in hps.items():
        setattr(args, k, v)
      exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
      if os.path.exists(exp_dir):
        continue
      print(hps)
      main(args)
  elif args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
      #print(','.join(str(i) for i, h in enumerate(all_hps) if h['opt'] == 'msgd_corr'))
      #print(' '.join(f'run_{i}' for i, h in enumerate(all_hps) if h['opt'] == 'msgd_corr'))
    else:
      hps = all_hps[args.run]
      print(hps)
      for k,v in hps.items():
        setattr(args, k, v)
      main(args)
  else:
    main(args)
