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

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
if 'SLURM_TMPDIR' in os.environ:
    print("Syncing locally")
    tmp_dir = os.environ['SLURM_TMPDIR'] + '/lztmp/'

    os.system(f"rsync -az {programs_dir} {tmp_dir}")
    os.system(f"rsync -az {datasets_dir} {tmp_dir}")
    programs_dir = f"{tmp_dir}/Programs"
    datasets_dir = f"{tmp_dir}/Datasets"
    print("Done syncing")
else:
    tmp_dir = "/tmp/lambdazero"

os.makedirs(tmp_dir, exist_ok=True)

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--nemb", default=32, help="#hidden", type=int)
parser.add_argument("--num_iterations", default=200000, type=int)
parser.add_argument("--num_conv_steps", default=12, type=int)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='atom_graph')
parser.add_argument("--model_version", default='v2')
parser.add_argument("--run", default=201, help="run", type=int)
#parser.add_argument("--save_path", default='/miniscratch/bengioe/LambdaZero/imitation_learning/')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--proxy_path", default='results/proxy__6/')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--progress", default='yes')






class Dataset:

    def __init__(self, bpath, device, repr_type):
        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(int(time.time()))
        self.train_mols = []
        self.test_mols = []
        self.train_mols_map = {}
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, repr_type)
        self.mdp.build_translation_table()
        self._device = device
        self.seen_molecules = set()
        self.stop_event = threading.Event()
        self.target_norm = [-8.6, 1.10]
        self.sampling_model = None
        self.sampling_model_prob = 0
        self.R_min = 1e-3

    def _get(self, i, dset):
        if (self.sampling_model_prob > 0 and # don't sample if we don't have to
            self.train_rng.uniform() < self.sampling_model_prob):
                return self._get_sample_model()
        # Sample trajectories by walking backwards from the molecules in our dataset
        m = dset[i]
        r = m.reward
        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action
        samples.append(((m,), ((-1, 0),), r, m, 1))
        while len(m.blocks): # and go backwards
            parents, actions = zip(*self.mdp.parents(m))
            samples.append((parents, actions, 0, m, 0))
            r = 0
            m = parents[self.train_rng.randint(len(parents))]
        return samples

    def set_sampling_model(self, model, proxy_reward, sample_prob=0.5):
        self.sampling_model = model
        self.sampling_model_prob = sample_prob
        self.proxy_reward = proxy_reward

    def _get_sample_model(self):
        m = BlockMoleculeDataExtended()
        r = 0
        done = 0
        samples = []
        for t in range(8): # TODO: max blocks config
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = self.sampling_model(s)
            m_p, s_p = self.sampling_model.out_to_policy(s, s_o, m_o)
            action = torch.cat([m_p.reshape(-1), s_p.reshape(-1)]).multinomial(1).item()
            if t > 2 and action == 0: # TODO: min blocks config as well
                done = 1
                action = (-1, 0)
                r = self._get_reward(m)
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m = self.mdp.add_block_to(m, *action)
                if not len(m.stems): # can't add anything more to this
                                     # mol so let's make it terminal
                    done = 1
                    r = self._get_reward(m)
            samples.append((*zip(*self.mdp.parents(m)), r, m, done))
            if done: break
        return samples

    def _get_reward(self, m):
        rdmol = m.mol
        if rdmol is None:
            return self.R_min
        smi = m.smiles
        if smi in self.train_mols_map:
            return self.train_mols_map[smi].reward
        return max(self.R_min, self.proxy_reward(m))

    def itertest(self, n):
        N = len(self.test_mols)
        for i in range(int(np.ceil(N/n))):
            samples = sum((self._get(j, self.test_mols) for j in range(i*n, min(N, (i+1)*n))), [])
            yield self.sample2batch(zip(*samples))

    def sample(self, n):
        eidx = self.train_rng.randint(0, len(self.train_mols), n)
        samples = sum((self._get(i, self.train_mols) for i in eidx), [])
        return zip(*samples)

    def sample2batch(self, mb):
        p, a, r, s, d, *o = mb
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []),
                               device=self._device).long()
        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch
        p = self.mdp.mols2batch(list(map(self.mdp.mol2repr, sum(p, ()))))
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).float()
        d = torch.tensor(d, device=self._device).float()
        return (p, p_batch, a, r, s, d, *o)

    def load_h5(self, path, args, test_ratio=0.05):
        import json
        import pandas as pd
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        store = pd.HDFStore(path, 'r')
        df = store.select('df')
        # Pandas has problem with calculating some stuff on float16
        df.dockscore = df.dockscore.astype("float64")
        for cl_mame in columns[2:]:
            df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)
        # Sample which indices will be our test set
        test_idxs = self.test_split_rng.choice(len(df), int(test_ratio * len(df)), replace=False)
        split_bool = np.zeros(len(df), dtype=np.bool)
        split_bool[test_idxs] = True
        for i in tqdm(range(len(df)), disable=not args.progress):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], df.iloc[i, c-1])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            # TODO: compute proper reward with QED & all
            m.reward = max(self.R_min,4-(min(0, m.dockscore)-self.target_norm[0])/self.target_norm[1])
            m.numblocks = len(m.blocks)
            if split_bool[i]:
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
                self.train_mols_map[df.iloc[i].name] = m
        store.close()


    def load_pkl(self, path, args, test_ratio=0.05):
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        mols = pickle.load(gzip.open(path))
        test_idxs = self.test_split_rng.choice(len(mols), int(test_ratio * len(mols)), replace=False)
        split_bool = np.zeros(len(mols), dtype=np.bool)
        split_bool[test_idxs] = True
        for i in tqdm(range(len(mols)), disable=not args.progress):
            m = BlockMoleculeDataExtended()
            for c in range(1, len(columns)):
                setattr(m, columns[c], mols[i][columns[c]])
            m.blocks = [self.mdp.block_mols[i] for i in m.blockidxs]
            # TODO: compute proper reward with QED & all
            m.reward = max(self.R_min,4-(min(0, m.dockscore)-self.target_norm[0])/self.target_norm[1])
            m.numblocks = len(m.blocks)
            if split_bool[i]:
                self.test_mols.append(m)
            else:
                self.train_mols.append(m)
                self.train_mols_map[m.smiles if len(m.blocks) else '[]'] = m




    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample(mbsize))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    self.sampler_threads[idx].failed = True
                    self.sampler_threads[idx].exception = e
                    self.ready_events[idx].set()
                    break
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [setattr(i, 'failed', False) for i in self.sampler_threads]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        if hasattr(self, 'sampler_threads'):
          while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]


def make_model(args, mdp):
    if args.repr_type == 'block_graph':
        model = model_block.GraphAgent(nemb=args.nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=1,
                                       num_conv_steps=args.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version='v4')#args.model_version)
    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=args.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=args.num_conv_steps,
                                     version=args.model_version)
    elif args.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    return model


class Proxy:
    def __init__(self, args, bpath, device):
        eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, eargs.repr_type)
        self.proxy = make_model(eargs, self.mdp)
        for a,b in zip(self.proxy.parameters(), params):
            a.data = torch.FloatTensor(b)
        self.proxy.to(device)

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()


def main(args):
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    device = torch.device('cuda')

    dataset = Dataset(bpath, device, args.repr_type)
    #dataset.load_h5("dock_db_1618610362tp_2021_04_16_17h.h5")
    dataset.load_h5("dock_db_1619111711tp_2021_04_22_13h.h5", args)
    dataset.load_pkl("small_mols.pkl.gz", args)
    dataset.load_pkl("small_mols_2.05.pkl.gz", args)

    exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
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
    model = make_model(args, mdp)
    model.to(device)

    proxy = Proxy(args, bpath, device)

    dataset.set_sampling_model(model, proxy, sample_prob=0.75)

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
            p, pb, a, r, s, d = r
        else:
            p, pb, a, r, s, d = dataset.sample2batch(dataset.sample(mbsize))
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        # state outputs
        stem_out_s, mol_out_s = model(s, None)
        # parents of the state outputs
        stem_out_p, mol_out_p = model(p, None)
        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        inflow = torch.log(torch.zeros((ntransitions,), device=device)
                           .index_add_(0, pb, torch.exp(qsa_p))) # pb is the parents' batch index
        # sum the state's Q(s,a), this is the outflow
        outflow = model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(r + outflow * (1-d))
        loss = (inflow - outflow_plus_r).pow(2).mean()
        # todo: reintroduce logsumexp with stem_o and mol_o above? or
        # have my own index_add_ stable version

        with torch.no_grad():
            term_loss = ((inflow - outflow_plus_r) * d).pow(2).sum() / (d.sum() + 1e-20)
            flow_loss = ((inflow - outflow_plus_r) * (1-d)).pow(2).sum() / ((1-d).sum() + 1e-20)
        loss.backward()
        last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
        train_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
        opt.step()
        opt.zero_grad()
        model.training_steps = i + 1
        #for _a,b in zip(model.parameters(), model_target.parameters()):
        #  b.data.mul_(1-tau).add_(tau*_a)

        #if any([torch.isnan(i).any() for i in model.parameters()]):
        if 0 and torch.isnan(loss):
            stop_event.set()
            stop_everything()
            raise ValueError()
            #pdb.set_trace()

        if not i % 100:
            #recent_best = np.argmax([i.rewards[0] * max(0,i.rewards[1]) * i.rewards[2] for i in replay.episodes[-100:]])
            #recent_best_m = np.max([i.rewards[0] * max(0,i.rewards[1]) * i.rewards[2] for i in replay.episodes[-100:]])
            #print(qsa)
            #print(r)
            #print(target)
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses)#, np.mean([i.gdist for i in replay.episodes[-100:]]),
            #recent_best_m, replay.episodes[-100+recent_best].rewards)
            #print(logit_reg)
            #print(stem_o.mean(), stem_o.max(), stem_o.min())
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

            if 0:
                qed_cutoff = [0.2, 0.7]
                synth_cutoff = [0, 0.4]
                all_rs = []
                for ep in replay.episodes[replay.num_loaded_episodes:]:
                    nrg, synth, qed = ep.rewards
                    qed_discount = (qed - qed_cutoff[0]) / (qed_cutoff[1] - qed_cutoff[0])
                    qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1
                    synth_discount = (synth - synth_cutoff[0]) / (synth_cutoff[1] - synth_cutoff[0])
                    synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1
                    all_rs.append(nrg * synth_discount * qed_discount)
                print(np.mean(all_rs), sorted(all_rs)[-10:])

            if i % 5000:
                continue

            if 1:
              save_stuff()
              continue

            t0 = time.time()
            total_test_loss = 0
            total_test_n = 0
            all_nlls = []
            all_ts = []
            for s, a, g, t in replay.iterate_test(max(mbsize, 128)):
                with torch.no_grad():
                    stem_o, mol_o = model(s, g)
                nlls = model.action_negloglikelihood(s, a, g, stem_o, mol_o)
                total_test_loss += nlls.sum().item()
                total_test_n += g.shape[0]
                all_nlls.append(nlls.data.cpu().numpy())
                all_ts.append(t)
            test_nll = total_test_loss / total_test_n
            if test_nll < best_test_loss:
              best_test_loss = test_nll
              best_model = deepcopy(model)
              best_model.to('cpu')
              early_stop_tol = max_early_stop_tolerance
            else:
              early_stop_tol -= 1
            print('test NLL:', test_nll)
            print('test took:', time.time() - t0)
            test_losses.append(test_nll)
            #test_infos.append((all_nlls, all_ts))
            save_stuff()
            if early_stop_tol <= 0:
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