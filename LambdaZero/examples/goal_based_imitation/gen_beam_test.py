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

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
if __name__ == "__main__":
  try:
    mp.set_start_method('spawn')
  except:
    pass
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


from LambdaZero.examples.goal_based_imitation import model_atom, model_block, model_fingerprint

from k_bins_test import MolMDPNC, ReplayBuffer, RolloutActor, MolEpisode

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



def generate_sampled_episodes(path, num_beam=200):
    args = pickle.load(gzip.open(f'{path}/info.pkl.gz'))['args']
    #if args.include_qed_data:
    #  return

    device = torch.device('cuda')
    print(synth_config)
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

    mdp = MolMDPNC(bpath)
    mdp.post_init(device, 'block_graph')

    print(args)
    model = model_block.GraphAgent(args.nemb, 3, mdp.num_blocks, 1, args.num_conv_steps, mdp,
                                   args.model_version)
    model.to(device)
    if osp.exists(f'{path}/best_params.pkl.gz'):
      loaded_params = pickle.load(gzip.open(f'{path}/best_params.pkl.gz'))
    else:
      loaded_params = pickle.load(gzip.open(f'{path}/params.pkl.gz'))

    with torch.no_grad():
      for p, lp in zip(model.parameters(), loaded_params):
        p.set_(torch.tensor(lp, device=device))

    replay = ReplayBuffer(bpath, device, 'block_graph')

    with replay._hashdb.begin(write=True) as replay._txn:
        past_ep = pickle.load(gzip.open('replays/random_10000.pkl.gz', 'rb'))
        past_ep += pickle.load(gzip.open('replays/random_10000_2.pkl.gz', 'rb'))
        replay.load_initial_episodes(past_ep, test_ratio=0.0)
        past_ep_qed = pickle.load(gzip.open('replays/random_qed_100000.pkl.gz', 'rb'))
        replay.load_initial_episodes(past_ep_qed, test_ratio=0)
    replay._txn = None

    qed_bit = 1 if args.include_qed_data else 0
    top_bin_goal2 = np.float32([[0]*(args.num_bins-1)+[1]+[0]*(args.num_bins-2)+[qed_bit,0]])
    stop_event = threading.Event()
    act = RolloutActor(bpath, device, 'block_graph', model,
                       stop_event, replay, synth_net, 'beam',
                       #set_goal=torch.tensor([[1,0,0]]).float().to(device))
                       set_goal=torch.tensor(top_bin_goal2).float().to(device),
                       dock_cpu_req=8)
    act.beam_max = num_beam
    print("starting beam search")
    #with replay._hashdb.begin(write=True) as replay._txn:
    act._beam_episode()
    beam = replay.episodes[replay.num_loaded_episodes:]
    pickle.dump(beam, gzip.open(f'{path}/beam_{num_beam}_f.pkl.gz', 'wb'))

import os
import os.path

print(sys.argv)
#generate_sampled_episodes(sys.argv[1])
#for i in range(12,16):
#for i in [7,8,9,12,11,12]:
#for i in [13,14,15,16,17,18]:
for i in range(64):
  exp_dir = f'results/array_feb_23_{i}/'
  if os.path.exists(f'{exp_dir}/beam_200_f.pkl.gz'):
    continue
  if os.path.exists(f'{exp_dir}/.lock'):
    continue
  with open(f'{exp_dir}/.lock', 'w') as f:
    f.write('lock')


  try:
    generate_sampled_episodes(exp_dir)
  except Exception as e:
    print(e)
  os.remove(f'{exp_dir}/.lock')
