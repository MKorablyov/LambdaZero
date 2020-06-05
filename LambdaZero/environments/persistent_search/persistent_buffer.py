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

from LambdaZero.environments.molecule import BlockMoleculeData
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


    def contains(self, mol):
        mol.blocks = [self.blocksd.block_mols[i] for i in mol.blockidxs]
        smi = Chem.MolToSmiles(mol.mol)
        if smi in self.smiles:
            return True
        return False

    def add(self, mol, mol_info):
        if len(self.mols) >= self.max_size:
            self.prune()
        mol.blocks = [self.blocksd.block_mols[i] for i in mol.blockidxs]
        smi = Chem.MolToSmiles(mol.mol)
        mol.blocks = None
        if smi in self.smiles:
            return
        self.mols.append((mol, mol_info['base_reward'], mol_info['QED'], smi))
        self.sumtree.set(len(self.mols)-1, np.exp(mol_info['base_reward'] / self.temperature))
        self.smiles.add(smi)

    def sample(self):
        idx = self.sumtree.sample(np.random.uniform(0,1))
        return self.mols[idx][0], self.mols[idx][1]

    def prune(self):
        new_mols_idx = np.argsort([i[1] for i in self.mols])[int(self.prune_factor * len(self.mols)):]
        new_sum_tree = SumTree(self.max_size)
        new_mols = []
        new_smiles = set()
        for i, j in enumerate(new_mols_idx):
            new_mols.append(self.mols[j])
            new_sum_tree.set(i, self.mols[j][1])
            new_smiles.add(self.mols[j][3])
        self.mols = new_mols
        self.sumtree = new_sum_tree
        self.smiles = new_smiles


class BlockMolEnv_PersistentBuffer:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems", "blockidxs"]

    def __init__(self, config):
        self.num_blocks = config["num_blocks"]
        self.max_blocks = config["max_blocks"]
        self.max_steps = config["max_steps"]
        self.max_simulations = config["max_simulations"]
        self.random_blocks = config["random_blocks"]
        #
        self.molMDP = MolMDP(**config["molMDP_config"])
        self.observ = FPObs_v1(config, self.molMDP)
        self.reward = config["reward"](**config["reward_config"])

        self.action_space = self.observ.action_space
        self.observation_space = self.observ.observation_space
        self.searchbuf = config['searchbuf']
        self.penalize_repeat = config.get('penalize_repeat', False)
        self.first_reset = True

    def _if_terminate(self):
        terminate = False
        # max steps
        if self.num_steps >= self.max_steps:
            terminate = True
        # max simulations
        if self.max_simulations is not None:
            if self.num_simulations >= self.max_simulations:
                terminate = True
        return terminate

    def reset(self):
        self.num_steps = 0
        self.num_simulations = 0
        if self.first_reset:
            self.first_reset = False
            self.molMDP.reset()
            self.molMDP.random_walk(self.random_blocks)
            last_reward = 0
        else:
            self.molMDP.molecule.blocks = None
            ray.get(self.searchbuf.add.remote(self.molMDP.molecule, self.last_reward_info))
            self.molMDP.reset()
            self.molMDP.molecule, last_reward = ray.get(self.searchbuf.sample.remote())
            self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx]
                                           for idx in self.molMDP.molecule.blockidxs]
            self.molMDP.molecule._mol = None
        self.reward.reset(last_reward)
        return self.observ(self.molMDP.molecule, self.num_steps)

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
        obs = self.observ(self.molMDP.molecule, self.num_steps)
        done = self._if_terminate()
        reward, info = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        self.last_reward_info = copy(info)
        info["molecule"] = self.molMDP.molecule
        if self.penalize_repeat and reward != 0:
            if ray.get(self.searchbuf.contains.remote(self.molMDP.molecule)):
                reward = 0 # exploration penalty
        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.molMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = deepcopy(self.num_steps)
        num_simulations = deepcopy(self.num_simulations)
        previous_reward = deepcopy(self.reward.previous_reward)
        mol = deepcopy(self.molMDP.molecule._mol)
        return mol_attr, num_steps, num_simulations, previous_reward, mol

    def set_state(self,state):
        mol_attr, self.num_steps, self.num_simulations, self.reward.previous_reward, self.molMDP.molecule._mol \
            = deepcopy(state)
        [setattr(self.molMDP.molecule, key, value) for key, value in mol_attr.items()]
        self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx] for idx in state[0]["blockidxs"]]
        self.molMDP.molecule._mol = None
        return self.observ(self.molMDP.molecule, self.num_steps)

    def render(self, outpath):
        mol = self.molMDP.molecule.mol
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)
