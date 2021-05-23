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
from rdkit import DataStructs
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
from train_proxy import Dataset as ProxyDataset
from main_flow import Dataset as GenModelDataset

import importlib
importlib.reload(model_atom)

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
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


parser = argparse.ArgumentParser()

parser.add_argument("--proxy_learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--proxy_dropout", default=0.1, help="MC Dropout in Proxy", type=float)
parser.add_argument("--proxy_weight_decay", default=1e-5, help="Weight Decay in Proxy", type=float)
parser.add_argument("--proxy_mbsize", default=64, help="Minibatch size", type=int)
parser.add_argument("--proxy_opt_beta", default=0.9, type=float)
parser.add_argument("--proxy_nemb", default=64, help="#hidden", type=int)
parser.add_argument("--proxy_num_iterations", default=100, type=int)
parser.add_argument("--num_init_examples", default=25000, type=int)
parser.add_argument("--num_outer_loop_iters", default=20, type=int)
parser.add_argument("--num_samples", default=32, type=int)
parser.add_argument("--proxy_num_conv_steps", default=12, type=int)
parser.add_argument("--proxy_repr_type", default='atom_graph')
parser.add_argument("--proxy_model_version", default='v2')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--cpu_req", default=8)
parser.add_argument("--progress", action='store_true')

# gen_model
parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=4, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.99, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=8, type=int)
parser.add_argument("--num_iterations", default=200, type=int)
parser.add_argument("--num_conv_steps", default=6, type=int)
parser.add_argument("--log_reg_c", default=1e-2, type=float)
parser.add_argument("--reward_exp", default=4, type=float)
parser.add_argument("--sample_prob", default=0, type=float)
parser.add_argument("--clip_grad", default=0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
# parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='atom_graph')
parser.add_argument("--model_version", default='v5')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--proxy_path", default='results/proxy__6/')



class Docker:
    def __init__(self, tmp_dir, cpu_req=2):
        self.target_norm = [-8.6, 1.10]
        self.dock = chem.DockVina_smi(tmp_dir) #, cpu_req=cpu_req)

    def eval(self, mol, norm=False):
        s = "None"
        try:
            s = Chem.MolToSmiles(mol)
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        if not norm:
            return r
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        return reward

    def __call__(self, m):
        return self.eval(m)


class Proxy:
    def __init__(self, args, bpath, device):
        self.args = args
        # eargs = pickle.load(gzip.open(f'{args.proxy_path}/info.pkl.gz'))['args']
        # params = pickle.load(gzip.open(f'{args.proxy_path}/best_params.pkl.gz'))
        self.mdp = MolMDPExtended(bpath)
        self.mdp.post_init(device, args.proxy_repr_type)
        self.mdp.floatX = torch.double
        self.proxy = make_model(args, self.mdp, is_proxy=True)
        # for a,b in zip(self.proxy.parameters(), params):
        #     a.data = torch.tensor(b, dtype=self.mdp.floatX)
        self.proxy.to(device)
        self.device = device
    
    def reset(self):
        for layer in self.proxy.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self, dataset):
        self.reset()
        stop_event = threading.Event()
        best_model = self.proxy
        best_test_loss = 1000
        opt = torch.optim.Adam(self.proxy.parameters(), self.args.learning_rate, betas=(self.args.opt_beta, 0.999),
                                weight_decay=self.args.proxy_weight_decay)

        debug_no_threads = False
        mbsize = self.args.mbsize

        if not debug_no_threads:
            sampler = dataset.start_samplers(8, mbsize)

        last_losses = []

        def stop_everything():
            stop_event.set()
            print('joining')
            dataset.stop_samplers_and_join()

        train_losses = []
        test_losses = []
        test_infos = []
        time_start = time.time()
        time_last_check = time.time()

        max_early_stop_tolerance = 5
        early_stop_tol = max_early_stop_tolerance

        for i in range(self.args.proxy_num_iterations+1):
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
            stem_out_s, mol_out_s = self.proxy(s, None, do_stems=False)
            loss = (mol_out_s[:, 0] - r).pow(2).mean()
            loss.backward()
            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))
            opt.step()
            opt.zero_grad()
            self.proxy.training_steps = i + 1

            if not i % 1000:
                last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
                print(i, last_losses)
                print('time:', time.time() - time_last_check)
                time_last_check = time.time()
                last_losses = []

                if i % 5000:
                    continue

                if 0:
                    # save_stuff()
                    continue

                t0 = time.time()
                total_test_loss = 0
                total_test_n = 0
                
                for s, r in dataset.itertest(max(mbsize, 128)):
                    with torch.no_grad():
                        stem_o, mol_o = self.proxy(s, None, do_stems=False)
                        loss = (mol_o[:, 0] - r).pow(2)
                        total_test_loss += loss.sum().item()
                        total_test_n += loss.shape[0]
                test_loss = total_test_loss / total_test_n
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_model = deepcopy(self.proxy)
                    best_model.to('cpu')
                    early_stop_tol = max_early_stop_tolerance
                else:
                    early_stop_tol -= 1
                    print('test loss:', test_loss)
                    print('test took:', time.time() - t0)
                    test_losses.append(test_loss)
                    # save_stuff()
                if early_stop_tol <= 0 and False:
                    print("Early stopping")
                    break
        
        stop_everything()
        self.proxy = deepcopy(best_model)
        self.proxy.to(self.device)
        print('Done.')

    def __call__(self, m):
        m = self.mdp.mols2batch([self.mdp.mol2repr(m)])
        return self.proxy(m, do_stems=False)[1].item()


def make_model(args, mdp, is_proxy=False):
    repr_type = args.proxy_repr_type if is_proxy else args.repr_type
    nemb = args.proxy_nemb if is_proxy else args.nemb
    num_conv_steps = args.proxy_num_conv_steps if is_proxy else args.num_conv_steps
    model_version = args.proxy_model_version if is_proxy else args.model_version
    if repr_type == 'block_graph':
        model = model_block.GraphAgent(nemb=nemb,
                                       nvec=0,
                                       out_per_stem=mdp.num_blocks,
                                       out_per_mol=1,
                                       num_conv_steps=num_conv_steps,
                                       mdp_cfg=mdp,
                                       version='v4')
    elif repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=args.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=args.num_conv_steps,
                                     version=model_version, 
                                     dropout_rate=args.proxy_dropout)
    elif repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
    return model


_stop = [None]
def train_generative_model(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = False
    device = torch.device('cuda')

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
                           betas=(args.opt_beta, args.opt_beta2))

    tf = lambda x: torch.tensor(x, device=device).double()

    mbsize = args.mbsize

    if not debug_no_threads:
        sampler = dataset.start_samplers(8, mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    _stop[0] = stop_everything

    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    log_reg_c = args.log_reg_c
    clip_loss = tf([args.clip_loss])

    for i in range(num_steps):
        if not debug_no_threads:
            r = sampler()
            for thread in dataset.sampler_threads:
                if thread.failed:
                    stop_everything()
                    pdb.post_mortem(thread.exception.__traceback__)
            p, pb, a, r, s, d = r
        else:
            p, pb, a, r, s, d = dataset.sample2batch(dataset.sample(mbsize))
        # Since we sampled 'mbsize' trajectories, we're going to get
        # roughly mbsize * H (H is variable) transitions
        ntransitions = r.shape[0]
        # state outputs
        if tau > 0:
            with torch.no_grad():
                stem_out_s, mol_out_s = target_model(s, None)
        else:
            stem_out_s, mol_out_s = model(s, None)
        # parents of the state outputs
        stem_out_p, mol_out_p = model(p, None)
        # index parents by their corresponding actions
        qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
        # then sum the parents' contribution, this is the inflow
        exp_inflow = (torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
                      .index_add_(0, pb, torch.exp(qsa_p))) # pb is the parents' batch index
        inflow = torch.log(exp_inflow + log_reg_c)
        # sum the state's Q(s,a), this is the outflow
        exp_outflow = model.sum_output(s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0]))
        # include reward and done multiplier, then take the log
        # we're guarenteed that r > 0 iff d = 1, so the log always works
        outflow_plus_r = torch.log(log_reg_c + r + exp_outflow * (1-d))
        losses = _losses = (inflow - outflow_plus_r).pow(2)
        if clip_loss > 0:
            ld = losses.detach()
            losses = losses / ld * torch.minimum(ld, clip_loss)

        term_loss = (losses * d).sum() / (d.sum() + 1e-20)
        flow_loss = (losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        loss = term_loss + flow_loss
        opt.zero_grad()
        loss.backward()

        _term_loss = (_losses * d).sum() / (d.sum() + 1e-20)
        _flow_loss = (_losses * (1-d)).sum() / ((1-d).sum() + 1e-20)
        last_losses.append((loss.item(), term_loss.item(), flow_loss.item()))
        train_losses.append((loss.item(), _term_loss.item(), _flow_loss.item(),
                             term_loss.item(), flow_loss.item()))
        train_infos.append((_term_loss.data.cpu().numpy(),
                            _flow_loss.data.cpu().numpy(),
                            exp_inflow.data.cpu().numpy(),
                            exp_outflow.data.cpu().numpy(),
                            r.data.cpu().numpy(),
                            ))
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(),
                                           args.clip_grad)
        opt.step()
        model.training_steps = i + 1
        if tau > 0:
            for _a,b in zip(model.parameters(), target_model.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)


        if not i % 100:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses)
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

            if not i % 1000 and do_save:
                save_stuff()

    stop_everything()
    if do_save:
        save_stuff()
    return model, dataset, test_infos


def sample_and_update_dataset(args, model, proxy_dataset, generator_dataset, docker):
    generator_dataset.set_sampling_model(model, docker, sample_prob=args.sample_prob)
    sampler = generator_dataset.start_samplers(8, args.num_samples)
    sampled_mols = sampler()
    dists =[]
    for m1, m2 in zip(sampled_mols, sampled_mols[1:] + sampled_mols[:1]):
        dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol), Chem.RDKFingerprint(m2.mol))
        dists.append(dist)
    
    rewards = []
    for m in sampled_mols:
        rewards.append(m.reward)
    proxy_dataset.add_samples(sampled_mols)
    return proxy_dataset, {'dists': dists, 'rewards': rewards }


def main(args):
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    device = torch.device('cuda')

    docker = Docker(tmp_dir, cpu_req=args.cpu_req)

    proxy_dataset = ProxyDataset(args, bpath, device, floatX=torch.float)
    proxy_dataset.load_h5("/scratch/mjain/dock_db_1619111711tp_2021_04_22_13h.h5", args, num_examples=args.num_init_examples)

    exp_dir = f'{args.save_path}/proxy_{args.array}_{args.run}/'
    os.makedirs(exp_dir, exist_ok=True)

    print(len(proxy_dataset.train_mols), 'train mols')
    print(len(proxy_dataset.test_mols), 'test mols')
    print(args)

    proxy = Proxy(args, bpath, device)    
    mdp = proxy_dataset.mdp
    
    proxy.train(proxy_dataset)

    for i in range(args.num_outer_loop_iters):
        print(f"Starting step: {i}")
        # Initialize model and dataset for training generator
        model = make_model(args, mdp)
        model.to(torch.double)
        model.to(device)
        gen_model_dataset = GenModelDataset(args, bpath, device)
        
        # train model with with proxy
        print(f"Training model: {i}")
        model, gen_model_dataset, training_metrics = train_generative_model(args, model, proxy, gen_model_dataset, do_save=False)

        # sample molecule batch for generator and update dataset with docking scores for sampled batch
        proxy_dataset, batch_metrics = sample_and_update_dataset(model, proxy_dataset, gen_model_dataset, docker)
        
        # update proxy with new data
        proxy.train(proxy_dataset)
    

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)