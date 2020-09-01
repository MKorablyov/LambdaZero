import os,time
from abc import ABC
import math
import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.models.tf.misc import normc_initializer

th, nn = try_import_torch()
from torch_geometric.nn import NNConv, Set2Set, GCNConv
import torch.nn.functional as F

from rdkit import Chem
import LambdaZero.environments
from LambdaZero import chem


def convert_to_tensor(arr):
	tensor = th.from_numpy(np.asarray(arr))
	if tensor.dtype == th.double:
		tensor = tensor.float()
	return tensor

class ActorCriticModel(TorchModelV2, nn.Module, ABC):
	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
		nn.Module.__init__(self)

		self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)

		self.shared_layers = None
		self.actor_layers = None
		self.critic_layers = None

		self._value_out = None

	def forward(self, input_dict, state, seq_lens):
		x = input_dict["obs"]
		x = self.shared_layers(x)
		# actor outputs
		logits = self.actor_layers(x)
		# compute value
		self._value_out = self.critic_layers(x)
		return logits, None

	def value_function(self):
		return self._value_out

	def compute_priors_and_value(self, obs):
		obs = convert_to_tensor([self.preprocessor.transform(obs)])
		input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

		with th.no_grad():
			model_out = self.forward(input_dict, None, [1])
			logits, _ = model_out
			value = self.value_function()
			logits, value = th.squeeze(logits), th.squeeze(value)
			priors = nn.Softmax(dim=-1)(logits)

			priors = priors.cpu().numpy()
			value = value.cpu().numpy()

			return priors, value



class MolActorCritic_thv1(TorchModelV2, nn.Module, ABC):
	def __init__(self, obs_space, action_space, num_outputs, model_config, name):
		TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
		nn.Module.__init__(self)

		self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)
		mol_fp_len = obs_space.original_space["mol_fp"].shape[0]
		stem_fp_len = obs_space.original_space["stem_fps"].shape[1]
		n_blocks = (obs_space.original_space["action_mask"].shape[0] -
					obs_space.original_space["jbond_fps"].shape[0] - 1) // obs_space.original_space["stem_fps"].shape[0]
		self.max_steps = obs_space.original_space["num_steps"].n

		self.shared_layers = nn.Sequential(
			nn.Linear(in_features=mol_fp_len + self.max_steps, out_features=256),nn.ReLU(),
			nn.Linear(in_features=256, out_features=256))
		self.stem_layers = nn.Sequential(
			nn.Linear(in_features=stem_fp_len, out_features=256))

		# build actor
		self.proc_stem = nn.Sequential(nn.Linear(in_features=stem_fp_len, out_features=256))
		self.actor_stop = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256, out_features=1))
		self.actor_add = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256*3, out_features=n_blocks))
		self.actor_break = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256*3, out_features=n_blocks))
		# build critic
		self.critic_layers = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256, out_features=1))
		self._value_out = None


	def forward(self, input_dict, state, seq_lens):
		# shared molecule embedding
		# weak todo (maksym) use mask before compute
		mol_fp = input_dict["mol_fp"]
		stem_fps = input_dict["stem_fps"]
		jbond_fps = input_dict["jbond_fps"]
		num_steps = input_dict["num_steps"]
		action_mask = input_dict["action_mask"]

		# shared layers
		mol_embed = self.shared_layers(th.cat([mol_fp, num_steps], 1))

		# actor outputs
		stop_logit = self.actor_stop(mol_embed)
		#
		jbond_embed = self.stem_layers(jbond_fps)
		mol_embed_ = mol_embed[:,None, :].repeat([1, jbond_embed.shape[1], 1])
		jbond_embed = th.cat([jbond_embed, mol_embed_, jbond_embed * mol_embed_], dim=2)
		break_logits = self.actor_break(jbond_embed)[:,:,0]
		#
		stem_embed = self.stem_layers(stem_fps)
		mol_embed_ = mol_embed[:,None, :].repeat([1, stem_embed.shape[1], 1])
		stem_embed = th.cat([stem_embed, mol_embed_, stem_embed * mol_embed_], dim=2)
		add_logits = self.actor_add(stem_embed)
		add_logits = add_logits.reshape([add_logits.shape[0], -1])
		#
		actor_logits = th.cat([stop_logit, break_logits, add_logits], axis=1)
		# compute value
		critic_logits = self.critic_layers(mol_embed)
		self._value_out = critic_logits[:,0]

		# mask not available actions
		masked_actions = (1. - action_mask).to(th.bool)
		actor_logits[masked_actions] = -20 # some very small prob that does not lead to inf
		return actor_logits, None

	def value_function(self):
		return self._value_out

	def compute_priors_and_value(self, obs):
		obs = th.tensor([self.preprocessor.transform(obs)]).float().cuda()
		input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

		with th.no_grad():
			model_out = self.forward(input_dict, None, [1])
			logits, _ = model_out
			value = self.value_function()
			logits, value = th.squeeze(logits), th.squeeze(value)
			priors = nn.Softmax(dim=-1)(logits)

			priors = priors.cpu().numpy()
			value = value.cpu().numpy()
			return priors, value

	def _save(self, checkpoint_dir):
		checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
		th.save(self.model.state_dict(), checkpoint_path)
		return checkpoint_path

	def _restore(self, checkpoint_path):
		self.model.load_state_dict(th.load(checkpoint_path))


class MPNNet(th.nn.Module):
	def __init__(self, num_feat=14, dim=64):
		super(MPNNet, self).__init__()
		self.lin0 = th.nn.Linear(num_feat, dim)

		h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
		self.conv = NNConv(dim, dim, h, aggr='mean')
		self.gru = nn.GRU(dim, dim)

		self.set2set = Set2Set(dim, processing_steps=3)
		self.lin1 = th.nn.Linear(2 * dim, dim)
		self.lin2 = th.nn.Linear(dim, 1)

	def forward(self, data):
		out = nn.functional.relu(self.lin0(data.x))
		h = out.unsqueeze(0)

		for i in range(3):
			m = nn.functional.relu(self.conv(out, data.edge_index, data.edge_attr))
			out, h = self.gru(m.unsqueeze(0), h)
			out = out.squeeze(0)

		out = self.set2set(out, data.batch)
		out = nn.functional.relu(self.lin1(out))
		out = self.lin2(out)
		return out.view(-1)
	

class ActorGCN(nn.Module):
	def __init__(self, n_features=14, n_hidden=64, n_blocks=105, n_stems=105, drop_prob=0.2):
		super(ActorGCN, self).__init__()
		# self.lin0 = th.nn.Linear(n_features, n_hidden)
		self.feature_dim = n_features
		mult = 1
		self.conv1 = GCNConv(n_features + 1, n_hidden, bias=True, normalize=True)
		self.conv2 = GCNConv(n_hidden, mult * n_hidden, bias=True, normalize=True)
		self.lin_blocks = nn.Linear(mult * n_hidden, n_blocks)
		self.lin_stems = nn.Linear(mult * n_hidden, n_stems)
		
		# self.conv21 = GCNConv(n_hidden, n_blocks, bias=True, normalize=True)
		# self.conv22 = GCNConv(n_hidden, n_blocks, bias=True, normalize=True)
		self.dropout_prob = drop_prob

	def forward(self, data):
		x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
		
		# concatenate stem_idxs to features
		stem_feature = th.zeros((x.size()[0]))
		stem_feature[data['stem_idxs']] = 1 #TODO: only works for batch size = 1
		stem_feature = stem_feature.unsqueeze(1)
		x = th.cat((x, stem_feature), dim=1)
		
		# # pass the concatenated features through the network
		# x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout_prob)
		# block_logits = self.conv21(x, edge_index)
		# stem_logits = self.conv22(x, edge_index)
		# #TODO: mask all the stem idxs other than the ones in the input
		# selected_block = F.gumbel_softmax(block_logits, tau=1, hard=True)
		# selected_stem = F.gumbel_softmax(stem_logits, tau=1, hard=True)
		# return block_logits, stem_logits, selected_block, selected_stem
		
		# pass the concatenated features through the network
		x = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout_prob)
		x = F.dropout(F.relu(self.conv2(x, edge_index)), p=self.dropout_prob)
		
		block_logits = self.lin_blocks(x)
		stem_logits = self.lin_stems(x)
		# TODO: mask all the stem idxs other than the ones in the input
		
		selected_block = F.gumbel_softmax(block_logits, tau=1, hard=False)
		selected_stem = F.gumbel_softmax(stem_logits, tau=1, hard=False)
		
		return block_logits, stem_logits, selected_block, selected_stem
	
class CriticGCN(nn.Module):
	def __init__(self, n_features=14, n_hidden=64, n_classes=105, drop_prob=0.2):
		super(CriticGCN, self).__init__()
		self.conv1 = GCNConv(n_features, n_hidden, bias=True, normalize=True)
		self.lin = nn.Linear(n_hidden, 1)
		self.dropout_prob = drop_prob
		
	def forward(self, mol_feature):
		docking_energy = F.dropout(F.relu(self.conv1(mol_feature)), p=self.dropout_prob)
		docking_energy = self.lin(docking_energy)
		return docking_energy


class GreedyActorCritic(nn.Module):
	def __init__(self, num_features, num_hidden, critic_hidden_multiplier, num_class):
		super(GreedyActorCritic, self).__init__()
		self.actor = ActorGCN(num_features, num_hidden, num_class)
		self.critic = CriticGCN(critic_hidden_multiplier, num_class)
		
		self.molMDP = LambdaZero.environments.MolMDP("/Users/sumanabasu/LambdaZero/datasets/fragdb/blocks_PDB_105.json")
		self.molMDP.reset()
		self.molMDP.random_walk(5)
	
	def forward(self, data):
		# Actor : Given a molecule, predicts the best block_idx and stem_idx
		block_logits, stem_logits, selected_block, selected_stem = self.actor.forward(data)
		
		best_block = th.argmax(th.max((block_logits + selected_block), dim=0).values)
		best_stem = th.argmax(th.max((stem_logits + selected_stem), dim=0).values)
		
		# Get Molecules by adding the predicted block at the predicted stem
		#TODO: make batch data compatible
		# molecules = [Chem.MolFromSmiles(smile) for smile in data['Smiles']]
		self.molMDP.molecule.mol = Chem.MolFromSmiles(data['smile'])
		# chem.mol_from_frag(frag_smis=, frags=self.blocks)
		self.molMDP.add_block(block_idx=best_block, stem_idx=best_stem)
		new_molecule = self.molMDP.molecule.mol # this is rdkit.Chem.rdchem.Mol object
		
		# # Get feature of the new molecule
		# molecule_feature =
		#
		#
		# # Critic
		# docking_energy = self.critic.forward(molecule_feature)
		#
		# return docking_energy