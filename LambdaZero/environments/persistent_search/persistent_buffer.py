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
from ray.rllib.utils import try_import_torch
torch, _ = try_import_torch()

import warnings
warnings.filterwarnings('ignore')

from LambdaZero.environments.molMDP import BlockMoleculeData
from LambdaZero.environments.block_mol_graph_v1 import BlockMolEnvGraph_v1
from LambdaZero.environments.block_mol_v3 import BlockMolEnv_v3
from LambdaZero.environments.block_mol_graph_v1 import BlockMolEnvGraph_v1
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
        self.mols = [(BlockMoleculeData(), -0.5, None, 0)]
        self.blocksd = BlocksData(config)
        self.max_size = config['max_size']
        self.prune_factor = config.get('prune_factor', 0.25)
        self.sumtree = SumTree(self.max_size)
        self.temperature = config.get('temperature', 2)
        self.threshold = config.get('threshold', 0.7)
        self.add_prob = config.get('add_prob', 0.5)
        self.smiles = set()
        self.mol_fps = []

    def contains(self, mol, threshold=None):
        if threshold is None:
            mol.blocks = [self.blocksd.block_mols[i] for i in mol.blockidxs]
            smi = Chem.MolToSmiles(mol.mol)
            if smi in self.smiles:
                return True
            return False
        else:
            if len(self.mol_fps) == 0:
                return False
            mol_fp = mol
            fp_buff = np.asarray(self.mol_fps)
            fp_buff = torch.FloatTensor(fp_buff).cuda()
            mol_fp = torch.FloatTensor(mol_fp).cuda()
            dist = torch.sum(torch.abs(fp_buff) - mol_fp, axis=1)
            dist = 1 - (dist/ torch.sum(torch.abs(fp_buff), 1) + torch.sum(torch.abs(mol_fp.view(-1, 512)), 1))
            # dist = np.sum(np.abs(fp_buff - mol_fp[None, :]), axis=1)
            # dist = 1 - (dist / (np.sum(np.abs(fp_buff),1) + np.sum(np.abs(mol_fp[None,:]),1)))
            return (dist > threshold).any().item()

    def add(self, mol, mol_info, mol_fp, use_similarity=True):
        if len(self.mols) >= self.max_size:
            self.prune()
        mol.blocks = [self.blocksd.block_mols[i] for i in mol.blockidxs]
        if use_similarity and self.contains(mol_fp, self.threshold):
            return len(self.mols)
        elif np.random.uniform(0, 1) > self.add_prob:
            return len(self.mols)
        smi = Chem.MolToSmiles(mol.mol)
        mol.blocks = None
        # if smi in self.smiles:
        #     return
        self.mols.append((mol, mol_info['reward'], smi, mol_fp))
        self.sumtree.set(len(self.mols) - 1, np.exp(mol_info['reward'] / self.temperature))
        self.smiles.add(smi)
        self.mol_fps.append(mol_fp)
        return len(self.mols)

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
            new_smiles.add(self.mols[j][2])
            new_mol_fps.append(self.mols[j][3])
        self.sumtree = new_sum_tree
        self.smiles = new_smiles
        self.mol_fps = new_mol_fps


class BlockMolEnv_PersistentBuffer(BlockMolEnv_v3):
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]

    def __init__(self, config=None):
        super().__init__(config)
        self.searchbuf = config['searchbuf']
        self.penalize_repeat = config.get('penalize_repeat', False)
        self.weight_exp_reward = config.get('weight_exp_reward', 0)
        self.first_reset = True
        self._reset = super().reset
        # self.ep_mol_fp = None

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
                ray.get(self.searchbuf.add.remote(self.molMDP.molecule, self.last_reward_info))
            self.molMDP.reset()
            self.molMDP.molecule, last_reward = ray.get(self.searchbuf.sample.remote())
            self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx]
                                           for idx in self.molMDP.molecule.blockidxs]
            self.molMDP.molecule._mol = None
        self.reward.reset(last_reward)
        obs, graph = self._make_obs()
        # self.ep_mol_fp = obs['mol_fp']
        return obs

    def _get_exp_reward(self, mol_fp):
        if self.ep_mol_fp is None:
            return 0
        dist = np.sum(np.abs(self.ep_mol_fp - mol_fp))
        dist = 1 - (dist / (np.sum(np.abs(self.ep_mol_fp)) + np.sum(np.abs(mol_fp))))
        return dist

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
        obs, graph = self._make_obs()
        done = self._if_terminate()
        reward, log_vals = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        self.last_reward_info = copy(log_vals)
        if (self.molMDP.molecule.mol is not None):
            smiles = Chem.MolToSmiles(self.molMDP.molecule.mol)
        else:
            smiles = None
        info = {"molecule": smiles, "log_vals": log_vals}
        done = any((simulate, done))
        # exp_reward = self._get_exp_reward(obs['mol_fp'])
        # info["molecule"] = self.molMDP.molecule
        if self.penalize_repeat and reward != 0:
            if ray.get(self.searchbuf.contains.remote(self.molMDP.molecule)):
                reward = 0 # exploration penalty
        return obs, reward, done, info

class BlockMolGraphEnv_PersistentBuffer(BlockMolEnvGraph_v1):
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]

    def __init__(self, config=None):
        super().__init__(config)
        self.first_reset = True
        self._reset = super().reset
        self.random_start_prob = config.get('random_start_prob', 0.75)
        #print(self.random_start_prob)
        self.episodes = 0

    def reset(self):
        self.num_steps = 0
        self.num_simulations = 0
        self.episodes += 1
        epsilon = np.random.uniform(0,1)
        if self.first_reset or epsilon < self.random_start_prob or self.episodes < 150:
            self.first_reset = False
            self._reset()
            last_reward = 0
        else:
            sampled_mol = ray.get(self.reward.reward_learner.sample.remote())
            if sampled_mol is None:
                self._reset()
                last_reward = 0
            else:
                self.molMDP.molecule.blocks = None
                self.molMDP.reset()
                
                self.molMDP.molecule = sampled_mol
                self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx]
                                            for idx in self.molMDP.molecule.blockidxs]
                self.molMDP.molecule._mol = None
        self.reward.reset(0)
        obs, graph = self._make_obs()
        return obs

class BlockMolGraphEnv_PersistentBuffer_(BlockMolEnvGraph_v1):
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]

    def __init__(self, config=None):
        super().__init__(config)
        obs_config = {"mol_fp_len": 512,
                  "mol_fp_radiis": [3],
                  "stem_fp_len": 64,
                  "stem_fp_radiis": [4, 3, 2]
                  }
        self.get_fps = chem.FPEmbedding_v2(**obs_config)
        self.searchbuf = config['searchbuf']
        self.penalize_repeat = config.get('penalize_repeat', False)
        self.first_reset = True
        self._reset = super().reset
        self.ep_mol_fp = None
        self.threshold = config.get('threshold', 0.7)
        self.random_start_prob = config.get('random_start_prob', 0)
        self.use_similarity = config.get('use_similarity', True)
        self.buffer_len = 0

    def reset(self):
        self.num_steps = 0
        self.num_simulations = 0
        epsilon = np.random.uniform(0,1)
        if self.first_reset or epsilon < self.random_start_prob:
            self.first_reset = False
            self._reset()
            last_reward = 0
        else:
            self.molMDP.molecule.blocks = None
            if len(self.last_reward_info.keys()) != 0:
                self.buffer_len = ray.get(self.searchbuf.add.remote(self.molMDP.molecule, 
                        self.last_reward_info, self.get_fps(self.molMDP.molecule)[0],
                        self.use_similarity))
            self.molMDP.reset()
            self.molMDP.molecule, last_reward = ray.get(self.searchbuf.sample.remote())
            self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx]
                                           for idx in self.molMDP.molecule.blockidxs]
            self.molMDP.molecule._mol = None
        self.reward.reset(last_reward)
        obs, graph = self._make_obs()
        # self.ep_mol_fp = obs['mol_fp']
        return obs

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
        obs, graph = self._make_obs()
        done = self._if_terminate()
        reward, log_vals = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        log_vals['reward'] = reward
        log_vals['buffer_size'] = self.buffer_len
        self.last_reward_info = copy(log_vals)
        if (self.molMDP.molecule.mol is not None):
            smiles = Chem.MolToSmiles(self.molMDP.molecule.mol)
        else:
            smiles = None
        info = {"molecule": smiles, "log_vals": log_vals}
        done = any((simulate, done))
        # if self.penalize_repeat and reward != 0:
        #     if ray.get(self.searchbuf.contains.remote(self.get_fps(self.molMDP.molecule)[0], self.threshold)):
        #         reward = 0 # exploration penalty
        return obs, reward, done, info