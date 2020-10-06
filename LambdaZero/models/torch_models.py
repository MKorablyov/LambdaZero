from abc import ABC
import os

import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch

# from torch_geometric.nn import GINEConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.nn import global_mean_pool


# import torch and torch.nn using ray utils
#torch, nn = try_import_torch()
import torch
from torch import nn
import torch.nn.functional as F
from .global_attention_layer import LowRankAttention


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from e3nn.point.message_passing import TensorPassingHomogenous
from e3nn.radial import GaussianRadialModel, CosineBasisModel, BesselRadialModel
from e3nn.point.gate import Gate
from e3nn.non_linearities.rescaled_act import swish, tanh, sigmoid, relu
from functools import partial


def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
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

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
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
            nn.Linear(in_features=mol_fp_len + self.max_steps, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256))
        self.stem_layers = nn.Sequential(
            nn.Linear(in_features=stem_fp_len, out_features=256))

        # build actor
        self.proc_stem = nn.Sequential(nn.Linear(in_features=stem_fp_len, out_features=256))
        self.actor_stop = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256, out_features=1))
        self.actor_add = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256 * 3, out_features=n_blocks))
        self.actor_break = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256 * 3, out_features=n_blocks))
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
        mol_embed = self.shared_layers(torch.cat([mol_fp, num_steps], 1))

        # actor outputs
        stop_logit = self.actor_stop(mol_embed)
        #
        jbond_embed = self.stem_layers(jbond_fps)
        mol_embed_ = mol_embed[:, None, :].repeat([1, jbond_embed.shape[1], 1])
        jbond_embed = torch.cat([jbond_embed, mol_embed_, jbond_embed * mol_embed_], dim=2)
        break_logits = self.actor_break(jbond_embed)[:, :, 0]
        #
        stem_embed = self.stem_layers(stem_fps)
        mol_embed_ = mol_embed[:, None, :].repeat([1, stem_embed.shape[1], 1])
        stem_embed = torch.cat([stem_embed, mol_embed_, stem_embed * mol_embed_], dim=2)
        add_logits = self.actor_add(stem_embed)
        add_logits = add_logits.reshape([add_logits.shape[0], -1])
        #
        actor_logits = torch.cat([stop_logit, break_logits, add_logits], axis=1)
        # compute value
        critic_logits = self.critic_layers(mol_embed)
        self._value_out = critic_logits[:, 0]

        # mask not available actions
        masked_actions = (1. - action_mask).to(torch.bool)
        actor_logits[masked_actions] = -20  # some very small prob that does not lead to inf
        return actor_logits, state

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        obs = torch.tensor([self.preprocessor.transform(obs)]).float().cuda()
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            return priors, value

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


class MPNNet(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, num_feat=14, dim=64):
        super(MPNNet, self).__init__()
        self.lin0 = nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)

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


class MPNNetDrop(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, drop_last, drop_data, drop_weights, drop_prob, num_feat=14, dim=64):
        super(MPNNetDrop, self).__init__()
        self.drop_last = drop_last
        self.drop_data = drop_data
        self.drop_weights = drop_weights
        self.drop_prob = drop_prob
        self.lin0 = nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.gru = nn.GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)

    def get_embed(self, data, do_dropout):
        if self.drop_data: data.x = F.dropout(data.x, training=do_dropout, p=self.drop_prob)
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        if self.drop_weights: h = F.dropout(h, training=do_dropout, p=self.drop_prob)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            if self.drop_weights: m = F.dropout(m, training=do_dropout, p=self.drop_prob)
            out, h = self.gru(m.unsqueeze(0), h)
            if self.drop_weights: h = F.dropout(h, training=do_dropout, p=self.drop_prob)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        if self.drop_weights: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
        out = F.relu(self.lin1(out))
        if self.drop_last: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
        return out

    def forward(self, data, do_dropout):
        embed = self.get_embed(data, do_dropout)
        out = self.lin2(embed)
        return out.view(-1)

    # todo: integrate masking
    # # if do_dropout and use_mask: out = out * self.set_mask
    # def set_mask(self):
    #     self.input_mask = torch.bernoulli(torch.zeros(self.num_feat).fill_(1 - self.drop_prob)).to(device)
    #     self.lin0_mask = torch.bernoulli(torch.zeros(self.dim).fill_(1 - self.drop_prob)).to(device)
    #     self.h_mask = torch.bernoulli(torch.zeros(self.dim).fill_(1 - self.drop_prob)).to(device)
    #     self.conv_mask = torch.bernoulli(torch.zeros(self.dim).fill_(1 - self.drop_prob)).to(device)
    #     self.gru_mask = torch.bernoulli(torch.zeros(self.dim).fill_(1 - self.drop_prob)).to(device)
    #     self.set_mask = torch.bernoulli(torch.zeros(self.dim + self.dim).fill_(1 - self.drop_prob)).to(device)
    #     self.lin1_mask = torch.bernoulli(torch.zeros(self.dim).fill_(1 - self.drop_prob)).to(device)
    #     self.lin2_mask = torch.bernoulli(torch.zeros(self.dim).fill_(1 - self.drop_prob)).to(device)
        
        
class MPNNetDropLRGA(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, drop_last, drop_data, drop_weights, drop_prob, num_feat=14, dim=64):
        super(MPNNetDropLRGA, self).__init__()
        self.drop_last = drop_last
        self.drop_data = drop_data
        self.drop_weights = drop_weights
        self.drop_prob = drop_prob
        self.lin0 = nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.lrga = nn.ModuleList([LowRankAttention(30, 14, drop_prob),
                     LowRankAttention(30, 14, drop_prob),
                     LowRankAttention(30, 14, drop_prob)])
        self.lrga_lin = nn.ModuleList([nn.Linear(60 + dim, dim),
                                       nn.Linear(60 + dim, dim),
                                       nn.Linear(60 + dim, dim)])

        self.gru = nn.GRU(dim+60, dim)
        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)

    def get_embed(self, data, do_dropout):
        if self.drop_data: data.x = F.dropout(data.x, training=do_dropout, p=self.drop_prob)
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        if self.drop_weights: h = F.dropout(h, training=do_dropout, p=self.drop_prob)

        for i in range(3):
            local_feat = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            if self.drop_weights: local_feat = F.dropout(local_feat, training=do_dropout, p=self.drop_prob)
            #with torch.no_grad():
            global_feat = 0* self.lrga[i](data.x)
            m = torch.cat([local_feat, global_feat], axis=1)
            #m = F.relu(self.lrga_lin[i](feat))
            #print("mean var", local_feat.mean(), local_feat.var(), global_feat.mean(), global_feat.var(),)

            out, h = self.gru(m.unsqueeze(0), h)
            if self.drop_weights: h = F.dropout(h, training=do_dropout, p=self.drop_prob)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        if self.drop_weights: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
        out = F.relu(self.lin1(out))
        if self.drop_last: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
        return out

    def forward(self, data, do_dropout):
        embed = self.get_embed(data, do_dropout)
        out = self.lin2(embed)
        return out.view(-1)


class GraphIsomorphismNet(nn.Module):
    def __init__(self,
                 node_feat: int = 14,
                 edge_feat: int = 4,
                 gin_layers: int = 2,
                 gin_size: int = 128,
                 gin_mlp_hidden: int = 128,
                 gin_hidden: int = 128,
                 linear_hidden: int = 128,
                 out_size: int = 1
                 ):
        """
        graph isomorphism network with a GRU / set2set pooling
        Args:
            node_feat (int, optional): number of input node features. Defaults to 14.
            edge_feat (int, optional): number of input edge features. Defaults to 4.
            gin_layers (int, optional): number of GIN layers. Defaults to 2.
            gin_size (int, optional): size of GCN embedding size. Defaults to 128.
            gin_mlp_hidden (int, optional): size of hidden layer in GIN MLP. Defaults to 128.
            gin_hidden (int, optional): output size of GIN. Defaults to 128.
            linear_hidden (int, optional): hidden size in the output fully-connected network. Defaults to 128.
            out_size (int, optional): output size. Defaults to 1.
        """
        super(GraphIsomorphismNet, self).__init__()

        self.node_lin0 = nn.Linear(node_feat, gin_size)
        self.edge_lin0 = nn.Linear(edge_feat, gin_size)

        gin_mlps = [nn.Sequential(
            nn.Linear(gin_size, gin_mlp_hidden),
            nn.ReLU(),
            nn.Linear(gin_mlp_hidden, gin_hidden))]

        for _ in range(gin_layers - 1):
            gin_mlps.append(nn.Sequential(
                nn.Linear(gin_hidden, gin_mlp_hidden),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden)))

        self.graphconv = nn.ModuleList([GINEConv(gmlp, train_eps=True) for gmlp in gin_mlps])

        # update edge features for each layer
        edge_mlps = [nn.Sequential(
            nn.Linear(gin_size, gin_mlp_hidden),
            nn.ReLU(),
            nn.Linear(gin_mlp_hidden, gin_hidden))]

        for _ in range(gin_layers - 1):
            edge_mlps.append(nn.Sequential(
                nn.Linear(gin_hidden, gin_mlp_hidden),
                nn.ReLU(),
                nn.Linear(gin_mlp_hidden, gin_hidden)))

        self.edge_mlps = nn.ModuleList(edge_mlps)

        self.gru = nn.GRU(gin_hidden, gin_hidden)

        self.set2set = Set2Set(gin_hidden, processing_steps=3)

        self.fully_connected = nn.Sequential(
            nn.Linear(2 * gin_hidden, linear_hidden),
            nn.ReLU(),
            nn.Linear(linear_hidden, out_size)
        )

    def forward(self, data):
        node_out = self.node_lin0(data.x)
        edge_out = self.edge_lin0(data.edge_attr)

        # graph convolutions
        h = node_out.unsqueeze(0)
        for gin, edge_update in zip(self.graphconv, self.edge_mlps):
            m = gin(node_out, data.edge_index, edge_out)
            node_out, h = self.gru(m.unsqueeze(0), h)
            node_out = node_out.squeeze(0)
            edge_out = edge_update(edge_out)

        # pooling with set2set
        node_out = self.set2set2(node_out, data.batch)
        # fully-connected layer for output
        out = self.fully_connected(node_out)
        return out.view(-1)


class TPNN_v1(TensorPassingHomogenous):
    def __init__(self, representations, use_set2set=True):
        hidden_size = representations[-1][0][0]
        self.use_set2set = use_set2set
        radial_model = partial(GaussianRadialModel, min_radius=0.7, max_radius=3.2, number_of_basis=10, h=100, L=3, act=swish)
        gate = partial(Gate, scalar_act=torch.tanh, tensor_act=torch.tanh)
        super().__init__(representations, radial_model, gate)
        if use_set2set:
            self.pooling = Set2Set(hidden_size, processing_steps=3)
            self.fully_connected = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.pooling = global_mean_pool
            self.fully_connected = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, graph):
        hidden_features = super().forward(graph)
        pooled_features = self.pooling(hidden_features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output


class TPNN_v2(TensorPassingHomogenous):
    def __init__(self, representations, **kwargs):
        emb_size = 16
        hidden_size = representations[-1][0][0]
        input_size = representations[0][0][0]
        representations[0][0] = (emb_size, 0, 0)
        radial_model = partial(GaussianRadialModel, min_radius=1.1, max_radius=2.3, number_of_basis=10, h=100, L=3, act=swish)
        gate = partial(Gate, scalar_act=torch.tanh, tensor_act=torch.tanh)
        super().__init__(representations, radial_model, gate)

        self.emb = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.Tanh()
        )
        self.pooling = Set2Set(hidden_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )

    def forward(self, graph):
        graph.x = self.emb(graph.x)
        hidden_features = super().forward(graph)
        pooled_features = self.pooling(hidden_features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output


class TPNN_v3(TensorPassingHomogenous):
    def __init__(self, representations, **kwargs):
        emb_size = 16
        hidden_size = representations[-1][0][0]
        input_size = representations[0][0][0]
        representations[0][0] = (emb_size, 0, 0)
        radial_model = partial(CosineBasisModel, max_radius=2.3, number_of_basis=10, h=100, L=3, act=swish)
        gate = partial(Gate, scalar_act=torch.tanh, tensor_act=torch.tanh)
        super().__init__(representations, radial_model, gate)

        self.emb = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.Tanh()
        )
        self.pooling = Set2Set(hidden_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )

    def forward(self, graph):
        graph.x = self.emb(graph.x)
        hidden_features = super().forward(graph)
        pooled_features = self.pooling(hidden_features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output


class TPNN_v4(TensorPassingHomogenous):
    def __init__(self, representations, **kwargs):
        emb_size = 16
        hidden_size = representations[-1][0][0]
        input_size = representations[0][0][0]
        representations[0][0] = (emb_size, 0, 0)
        radial_model = partial(BesselRadialModel, max_radius=2.3, number_of_basis=10, h=100, L=3, act=swish)
        gate = partial(Gate, scalar_act=relu, tensor_act=relu)
        super().__init__(representations, radial_model, gate)

        self.emb = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.Tanh()
        )
        self.pooling = Set2Set(hidden_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )

    def forward(self, graph):
        graph.x = self.emb(graph.x)
        hidden_features = super().forward(graph)
        pooled_features = self.pooling(hidden_features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output