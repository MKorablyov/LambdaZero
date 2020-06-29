# molecular environment

# molMDP().random_walk() -> mol_graph, r-groups
# choose r-group randomly
# molMDP().make_subs(r-group) -> mol_graph * 105
# MPNN(mol_graph, r_group) -> logit * 105           # actor
# MPNN(mol_graphs) -> label * 105                   # critic

# Read Dataset


# Actor : Maps
import LambdaZero.utils

from LambdaZero.inputs import inputs_op

import numpy as np
import os.path as osp
import pandas as pd
import ray
import torch
import torch.nn as nn

from ray import tune
from torch_geometric.data import (InMemoryDataset, DataLoader)

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
# transform = LambdaZero.utils.Complete()

CONFIG = {
	'trainer_config' : {
		'dataset_root' : osp.join(datasets_dir, "actor_dock/results"),
		'file_names' : ['actor_dock'],
		'transform' : LambdaZero.utils.Complete(),
		'split_name' : 'randsplit_ActorDock_walk_5',
		'b_size' : 5,
		'molprops' : ['initial_energry', 'stem_idxs', 'docking_energies']
	}
}
	
# class Actor(nn.Module):



class ActorCritic(tune.Trainable):
	def _setup(self, config):

		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


	# 	# make model
	# 	self.actor = LambdaZero.models.MPNNet()
	# 	self.actor.to(self.device)
	# 	self.optim = torch.optim.Adam(self.actor.parameters(), lr=config["lr"])
	#
	# 	self.critic =
	#
	# 	# make epochs
	# 	self.train_epoch = config["train_epoch"]
	# 	self.eval_epoch = config["eval_epoch"]
	#
	# def _train(self):
	# 	train_scores = self.train_epoch(self.train_set, self.model, self.optim, self.device, self.config)
	# 	eval_scores = self.eval_epoch(self.train_set, self.model,  self.device, self.config)
	# 	# rename to make scope
	# 	train_scores = [("train_" + k, v) for k,v in train_scores.items()]
	# 	eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
	# 	scores = dict(train_scores + eval_scores)
	# 	return scores
	#
	# def _save(self, checkpoint_dir):
	# 	checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
	# 	th.save(self.model.state_dict(), checkpoint_path)
	# 	return checkpoint_path


if __name__ == '__main__':
	ray.init()
	
	config = CONFIG['trainer_config']
	# load dataset
	dataset = inputs_op.BrutalDock(config['dataset_root'],
						props=config['molprops'],
						transform=config['transform'],
						file_names=config['file_names'])
	
	# split dataset
	
	split_path = osp.join(config['dataset_root'], config['split_name'] + ".npy")
	train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)
	train_set = DataLoader(dataset[torch.tensor(train_idxs)], shuffle=True, batch_size=config['b_size'])
	val_set = DataLoader(dataset[torch.tensor(val_idxs)], batch_size=CONFIG['b_size'])
	# test_set = DataLoader(dataset[torch.tensor(test_idxs)], batch_size=CONFIG['b_size'])
 

