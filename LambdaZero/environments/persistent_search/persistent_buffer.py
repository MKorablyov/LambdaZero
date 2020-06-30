from copy import copy, deepcopy
from collections import defaultdict
import gc
import gzip
import os
import os.path as osp
import pickle
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import ray
from rdkit import Chem
from rdkit.Chem import QED


import warnings
warnings.filterwarnings('ignore')

from LambdaZero.environments.molMDP import BlockMoleculeData
from LambdaZero.environments.block_mol_v3 import BlockMolEnv_v3
from LambdaZero.environments.persistent_search.fast_sumtree import SumTree

from LambdaZero import chem


class BlocksData:
    def __init__(self, config):
        blocks = pd.read_json(config['blocks_file'])
        self.block_smi = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in blocks["block_smi"]]
        self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
        self.num_blocks = len(self.block_smi)


class PersistentSearchBuffer:
    def __init__(self, config):
        self.mols = [(BlockMoleculeData(), -0.5, 0)]
        self.blocksd = BlocksData(config)
        self.max_size = config['max_size']
        self.prune_factor = config.get('prune_factor', 0.25)
        self.sumtree = SumTree(self.max_size)
        self.temperature = config.get('temperature', 2)
        self.smiles = set()
        self.mol_fps = []

    def contains(self, mol):
        mol.blocks = [self.blocksd.block_mols[i] for i in mol.blockidxs]
        smi = Chem.MolToSmiles(mol.mol)
        if smi in self.smiles:
            return True
        return False

    def distance(self, mol_fp):
        if len(self.mol_fps) == 0:
            return 0
        mol_buff = np.asarray(self.mol_fps)
        dist = np.sum(np.abs(mol_buff - mol_fp[None, :]), axis=1)
        dist = 1 - (dist / (np.sum(np.abs(mol_buff),1) + np.sum(np.abs(mol_fp[None,:]),1)))
        return np.mean(dist)

    def add(self, mol, mol_info, mol_fp):
        if len(self.mols) >= self.max_size:
            self.prune()
        mol.blocks = [self.blocksd.block_mols[i] for i in mol.blockidxs]
        smi = Chem.MolToSmiles(mol.mol)
        mol.blocks = None
        if smi in self.smiles:
            return
        self.mol_fps.append(mol_fp)
        self.mols.append((mol, mol_info['reward'], mol_info['qed'], smi, mol_fp))
        self.sumtree.set(len(self.mols)-1, np.exp(mol_info['reward'] / self.temperature))
        self.smiles.add(smi)

    def sample(self):
        idx = self.sumtree.sample(np.random.uniform(0,1))
        return self.mols[idx][0], self.mols[idx][1]

    def prune(self):
        new_mols_idx = np.argsort([i[1] for i in self.mols])[int(self.prune_factor * len(self.mols)):]
        new_sum_tree = SumTree(self.max_size)
        new_mols = []
        new_mol_fps = []
        new_smiles = set()
        for i, j in enumerate(new_mols_idx):
            new_mols.append(self.mols[j])
            new_sum_tree.set(i, self.mols[j][1])
            new_smiles.add(self.mols[j][3])
            new_mol_fps.append(self.mol_fps[j][4])
        self.mols = new_mols
        self.sumtree = new_sum_tree
        self.smiles = new_smiles
        self.mol_fps = new_mol_fps


class BlockMolEnv_PersistentBuffer(BlockMolEnv_v3):
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]

    def __init__(self, config=None):
        super().__init__(config)
        self.searchbuf = config['searchbuf']
        self.penalize_repeat = config.get('penalize_repeat', False)
        self.first_reset = True
        self._reset = super().reset

    def reset(self):
        self.num_steps = 0
        self.num_simulations = 0
        if self.first_reset:
            self.first_reset = False
            self._reset()
            last_reward = 0
        else:
            self.molMDP.molecule.blocks = None
            if len(self.last_reward_info.keys()) != 0:
                ray.get(self.searchbuf.add.remote(self.molMDP.molecule, self.last_reward_info, self.get_fps(self.molMDP.molecule)[0]))
            self.molMDP.reset()
            self.molMDP.molecule, last_reward = ray.get(self.searchbuf.sample.remote())
            import pdb; pdb.set_trace
            self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx]
                                           for idx in self.molMDP.molecule.blockidxs]
            self.molMDP.molecule._mol = None
        self.reward.reset(last_reward)
        return self._make_obs()

    def step(self, action):
        try:
            state = self.get_state()
            if (action == 0):
                simulate = True
                self.num_simulations += 1
            elif action <= (self.max_blocks - 1):
                simulate = False
                self.molMDP.remove_jbond(jbond_idx=action-1)
            else:
                simulate = False
                stem_idx = (action - self.max_blocks)//self.num_blocks
                block_idx = (action - self.max_blocks) % self.num_blocks
                self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)
        except Exception as e:
            print("error taking action", action)
            print(e)
            _mol = self.molMDP.molecule.mol
            print(Chem.MolToSmiles(_mol) if _mol else "no mol")
            print(self.get_state())
            with open('blockmolenv_step_error.txt', 'a') as f:
                print("error taking action", action, file=f)
                print(e, file=f)
                print(Chem.MolToSmiles(_mol) if _mol else "no mol", file=f)
                print(state, file=f)
                print(self.get_state(), file=f)
            self.set_state(state)

        self.num_steps += 1
        obs = self._make_obs()
        done = self._if_terminate()
        reward, log_vals = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        self.last_reward_info = copy(log_vals)
        info = {"molecule" : self.molMDP.molecule, "log_vals": log_vals}
        # import pdb;pdb.set_trace();
        done = any((simulate, done))
        exp_reward = ray.get(self.searchbuf.distance.remote(obs['mol_fp']))
        # info["molecule"] = self.molMDP.molecule
        if self.penalize_repeat and reward != 0:
            if ray.get(self.searchbuf.contains.remote(self.molMDP.molecule)):
                reward = 0 # exploration penalty
        return obs, reward + 1 * exp_reward, done, info
