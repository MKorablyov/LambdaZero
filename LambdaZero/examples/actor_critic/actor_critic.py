# molecular environment

# molMDP().random_walk() -> mol_graph, r-groups
# choose r-group randomly
# molMDP().make_subs(r-group) -> mol_graph * 105
# MPNN(mol_graph, r_group) -> logit * 105           # actor
# MPNN(mol_graphs) -> label * 105                   # critic

# Read Dataset


# Actor : Maps
import LambdaZero.models
import LambdaZero.utils

from LambdaZero.inputs import inputs_op
from LambdaZero.chem import mol_to_graph

import numpy as np
import os.path as osp
import pandas as pd
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import tune
from torch_geometric.data import (InMemoryDataset, DataLoader)
from torch_geometric.nn import GCNConv

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
# transform = LambdaZero.utils.Complete()

CONFIG = {
	'trainer_config' : {
		'dataset_root' : osp.join(datasets_dir, "actor_dock"),
		'file_names' : ['actor_dock'],
		'transform' : LambdaZero.utils.Complete(),
		'split_name' : 'randsplit_ActorDock_walk_5',
		'b_size' : 1,
		'molprops' : ['initial_energy', 'stem_idxs', 'docking_energies'],
		'num_features' : 14,
		'num_hidden' : 64, #tune.grid_search([16, 64, 256]),
		'num_class' : 105,
		'critic_hidden_multiplier' : 2, #tune.grid_search([2, 4]),
		'lr_actor' : 0.0001, #tune.grid_search([0.001, 0.0001, 0.00001]),
		'lr_critic' : 0.00001,
		'targets' : ['docking_energies']
	}
}
	
def train_epoch(loader, model, optimizer, device, config):
	model.train()

	metrics = {"loss":0, "mse": 0, "mae":0}
	for idx,data in enumerate(loader):
		# compute y_hat and y
		data = data.to(device)

		optimizer.zero_grad()
		pred_energy = model(data)
		target = data['docking_energies'][0]
		
		# print('pred:',pred_energy, 'size:', pred_energy.size())
		print('target:',target, 'size:', target.size())

		loss = F.mse_loss(pred_energy, target)
		loss.backward()
		optimizer.step()

		metrics["loss"] += loss.item() * data.num_graphs
		metrics["mse"] += ((target - pred_energy) ** 2).sum().item()
		metrics["mae"] += ((target - pred_energy).abs()).sum().item()

	metrics["loss"] = metrics["loss"] / len(loader.dataset)
	metrics["mse"] = metrics["mse"] / len(loader.dataset)
	metrics["mae"] = metrics["mae"] / len(loader.dataset)
	return metrics


class ActorCriticTrainer(tune.Trainable):
	def _setup(self, config):
		config = CONFIG['trainer_config']
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
		# load dataset
		dataset = inputs_op.ActorDoc(config['dataset_root'],
									 props=config['molprops'],
									 transform=config['transform'],
									 file_names=config['file_names'])

		# split dataset
		split_path = osp.join(config['dataset_root'], 'raw', config['split_name'] + ".npy")
		train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)
		self.train_set = DataLoader(dataset[torch.tensor(train_idxs)], shuffle=True, batch_size=config['b_size'])
		self.val_set = DataLoader(dataset[torch.tensor(val_idxs)], batch_size=config['b_size'])
		# self.test_set = DataLoader(dataset[torch.tensor(test_idxs)], batch_size=CONFIG['b_size'])

		# make model
		self.model = LambdaZero.models.GreedyActorCritic(config['num_features'],
														 config['num_hidden'],
														 config['critic_hidden_multiplier'],
														 config['num_class'])
		self.model.to(self.device)
		
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr_actor"])
		
		# # train actor and critic with different learning rates
		# self.optim_actor = torch.optim.Adam(self.model.actor.parameters(), lr=config["lr_actor"])
		# self.optim_critic = torch.optim.Adam(self.model.critic.parameters(), lr=config["lr_critic"])


	def _train(self):
		train_scores = train_epoch(self.train_set, self.model, self.optimizer, self.device, self.config)
		# eval_scores = eval_epoch(self.val_set, self.model,  self.device, self.config)
		
		# rename to make scope
		train_scores = [("train_" + k, v) for k,v in train_scores.items()]
		eval_scores = [("eval_" + k, v) for k, v in eval_scores.items()]
		scores = dict(train_scores + eval_scores)
		return scores

	def _save(self, checkpoint_dir):
		checkpoint_path = osp.join(checkpoint_dir, "model.pth")
		torch.save(self.model.state_dict(), checkpoint_path)
		return checkpoint_path

	def _restore(self, checkpoint_path):
		self.model.load_state_dict(torch.load(checkpoint_path))


if __name__ == '__main__':
	ray.init()
	
	# analysis = tune.run(ActorCriticTrainer,
	# 					name='GreedyActor',
	# 					config=CONFIG['trainer_config'],
	# 					stop={"training_iteration": 100},  # EarlyStop(),
	# 					resources_per_trial={
	# 						"cpu": 4 #TODO: set gpu before running on the cluster
	# 					},
	# 					num_samples=1,
	# 					local_dir='~/ray_results/test'
	# 					)
	
	config = CONFIG['trainer_config']
	# load dataset
	dataset = inputs_op.ActorDoc(config['dataset_root'],
								   props=config['molprops'],
								   transform=config['transform'],
								   file_names=config['file_names'])

	# split dataset

	split_path = osp.join(config['dataset_root'], 'raw', config['split_name'] + ".npy")
	train_idxs, val_idxs, test_idxs = np.load(split_path, allow_pickle=True)
	train_set = DataLoader(dataset[torch.tensor(train_idxs)], shuffle=True, batch_size=config['b_size'])
	# val_set = DataLoader(dataset[torch.tensor(val_idxs)], batch_size=config['b_size'])
	# test_set = DataLoader(dataset[torch.tensor(test_idxs)], batch_size=CONFIG['b_size'])
	#
	model = LambdaZero.models.GreedyActorCritic(config['num_features'],
													 config['num_hidden'],
													 config['critic_hidden_multiplier'],
													 config['num_class'])
	for idx, batch in enumerate(train_set):
		model.forward(batch)

