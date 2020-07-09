"""
This script generates a dataset at molecule level containing:
	-- smiles
	-- stem ids
	-- docking energy of all the blocks, for all stem ids, given a molecule
"""

import LambdaZero.environments
import LambdaZero.inputs
import LambdaZero.utils

import numpy as np
import os
import os.path as osp
import pandas as pd
import ray
import time
import torch

from copy import deepcopy
from rdkit import Chem
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip, Data, DataLoader)


class cfg:
	datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
	
	num_cpus = 4
	num_gpus = 1
	device = "cuda"
	
	db_name = "actor_dock"
	db_path = osp.join(datasets_dir, db_name)
	if not osp.exists(db_path):
		os.mkdir(db_path)
	results_dir = osp.join(datasets_dir, db_name, "raw")
	if not osp.exists(results_dir):
		os.mkdir(results_dir)
	
	# env parameters
	blocks_file = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
	
	# MPNN parameters
	dockscore_model = osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002")
	
	# dataset size
	dataset_len = 100000
	
	# data parallel
	num_workers = 2
	
	# Random walk lengths
	walk_length = 5
	
	# Split config
	# dataset_root = osp.join(self.results_dir, self.db_name + ".feather")
	# file_names = self.db_name + ".feather"
	split_name = "randsplit_ActorDock_walk_" + str(walk_length)
	split_probs = [0.8, 0.1, 0.1]


@ray.remote(num_gpus=cfg.num_gpus)
class Worker:
	def __init__(self):
		# initialize MDP
		self.molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=cfg.blocks_file)
		# Initialize docking reward prediction
		self.comp_reward = LambdaZero.environments.reward.PredDockReward(load_model=cfg.dockscore_model,
																		 natm_cutoff=[45, 50],
																		 qed_cutoff=[0.2, 0.7],
																		 soft_stop=False,
																		 exp=None,
																		 delta=False,
																		 simulation_cost=0.0,
																		 device=cfg.device)
		self.walk_length = cfg.walk_length
	
	def _sample(self):
		'''
		:return: 	smile of a randomly selected molecule
					stem indices where new molecules can be added
					docking energy of all the blocks, for all stem ids, given a molecule
		'''
		# do random walk
		self.molMDP.reset()
		self.comp_reward.reset()
		self.molMDP.random_walk(self.walk_length)
		
		init_molecule = deepcopy(self.molMDP.molecule)
		# iterate over places to change
		action_values = []
		for atmidx in self.molMDP.molecule.stem_atmidxs:
			# iterate over alphabet of building blocks
			addblock_values = []
			for block_idx in range(self.molMDP.num_blocks):
				self.molMDP.add_block(block_idx=block_idx, atmidx=atmidx)
				reward = self.comp_reward(self.molMDP.molecule, env_stop=False, simulate=True, num_steps=1)[0]
				addblock_values.append(reward)
				self.molMDP.molecule = deepcopy(init_molecule)
			action_values.append(addblock_values)
			
		return(Chem.MolToSmiles(self.molMDP.molecule.mol),
			   self.comp_reward(self.molMDP.molecule, env_stop=False, simulate=True, num_steps=1)[0],
			  self.molMDP.molecule.stem_atmidxs,
			   action_values)
	
	#TODO request that molecule object should provide torch graph and smiles


def generate_dataset():
	'''
	Parallel data generation unit
	:return: stores the generated dataset to disk
	'''
	ray.init()
	# time.sleep(60)
	
	workers = [Worker.remote() for i in range(cfg.num_workers)]
	tasks = [worker._sample.remote() for worker in workers]
	
	task2worker = {task: worker for task, worker in zip(tasks, workers)}
	
	samples = []
	for i in range(cfg.dataset_len):
		print('creating data ',i)
		done_task, tasks = ray.wait(tasks)
		done_worker = task2worker.pop(done_task[0])
		samples.extend(ray.get(done_task))
		
		new_task = done_worker._sample.remote()
		task2worker[new_task] = done_worker
		tasks.append(new_task)
		
		if (i % 10) == 0:
			df = pd.DataFrame(samples, columns=('smile', 'initial_energry', 'stem_idxs', 'docking_energies'))
			df.to_parquet(osp.join(cfg.results_dir, cfg.db_name) + ".feather")
	
	df = pd.DataFrame(samples, columns=('smile', 'initial_energry', 'stem_idxs', 'docking_energies'))
	df.to_feather(osp.join(cfg.results_dir, cfg.db_name) + ".feather")
	

def split_dataset():
	'''
	splits the dataset int train, validation and test sets
	:return: stores the indices of each set as a list of [train_idx, valid_idx, test_idx]
	'''
	data = pd.read_parquet(osp.join(cfg.results_dir, cfg.db_name) + ".feather")
	splits = LambdaZero.inputs.random_split(len(data), cfg.split_probs)

	split_path = osp.join(cfg.results_dir, cfg.split_name + ".npy")
	np.save(split_path, splits)
	
	
if __name__ == "__main__":
	generate_dataset()
	split_dataset()


