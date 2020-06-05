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
#from LambdaZero.examples.config import datasets_dir


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

@ray.remote
class SimDockRewardActor:

    def __init__(self, tmp_dir, programs_dir, datasets_dir, num_threads=1):
        self.dock = chem.Dock_smi(tmp_dir,
                                  osp.join(programs_dir, 'chimera'),
                                  osp.join(programs_dir, 'dock6'),
                                  osp.join(datasets_dir, 'brutal_dock/d4/docksetup/'))
        self.target_norm = [-26.3, 12.3]
        self.running = True
        self.num_threads = num_threads

    def run(self, tree, n):
        while self.running:
            self.do_iterations(tree, n)

    def stop(self):
        self.running = False

    def do_iterations(self, tree, n):
        mols = ray.get(tree.get_top_k_nodes.remote(n, sim_dock=True))
        # mols is a list of BlockMoleculeData objects
        mols = [i for i in mols if i[2] is not None]
        n = len(mols)
        rewards = [None] * n
        idxs = [None] * n
        def f(i, idx):
            try:
                _, r, _ = self.dock.dock(Chem.MolToSmiles(mols[i][2].mol))
            except Exception as e: # Sometimes the prediction fails
                print(e)
                r = 0
            rewards[i] = -(r-self.target_norm[0])/self.target_norm[1]
            idxs[i] = idx
        t0 = time.time()
        threads = []
        thread_at = self.num_threads
        for i, (_, idx, mol) in enumerate(mols):
            threads.append(threading.Thread(target=f, args=(i, idx)))
            if i < self.num_threads:
                threads[-1].start()
        while None in rewards:
            if sum([i.is_alive() for i in threads]) < self.num_threads and thread_at < n:
                threads[thread_at].start()
                thread_at += 1
            time.sleep(0.5)
        t1 = time.time()
        print(f"Ran {n} docking simulations in {t1-t0:.2f}s ({(t1-t0)/n:.2f}s/mol)")
        tree.set_sim_dock_reward.remote(idxs, rewards)
