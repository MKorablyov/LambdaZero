from abc import ABC
import os, time

import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch

#from torch_geometric.nn import GINEConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set


# import torch and torch.nn using ray utils
#torch, nn = try_import_torch()
import torch
from torch import nn
import torch.nn.functional as F

import os.path as osp
from torch.nn import Linear, Embedding
from torch_scatter import scatter
from torch_sparse import SparseTensor
from torch_geometric.nn import radius_graph
from torch_geometric.data import download_url
from torch_geometric.data.makedirs import makedirs

from LambdaZero.models import bessel_basis, real_sph_harm
from math import sqrt, pi as PI

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

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a**2) * fan))
        tensor.data.uniform_(-bound, bound)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def normal(tensor, mean, std):
    if tensor is not None:
        tensor.data.normal_(mean, std)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def swish(x):
    return x * x.sigmoid()

try:
    import sympy as sym
except ImportError:
    sym = None

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
except ImportError:
    tf = None

qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return 1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super(BesselBasisLayer, self).__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super(SphericalBasisLayer, self).__init__()
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish):
        super(EmbeddingBlock, self).__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_bilinear, num_spherical,
                 num_radial, num_before_skip, num_after_skip, act=swish):
        super(InteractionBlock, self).__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear,
                              bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.Tensor(hidden_channels, num_bilinear, hidden_channels))

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_channels, num_layers,
                 act=swish):
        super(OutputBlock, self).__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNet(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels, num_blocks, num_bilinear,
                 num_spherical, num_radial, cutoff=5.0, envelope_exponent=5,
                 num_before_skip=1, num_after_skip=2, num_output_layers=3,
                 act=swish):
        super(DimeNet, self).__init__()

        self.cutoff = cutoff

        if sym is None:
            raise ImportError('Package `sympy` could not be found.')

        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(num_radial, hidden_channels, out_channels,
                        num_output_layers, act) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(hidden_channels, num_bilinear, num_spherical,
                             num_radial, num_before_skip, num_after_skip, act)
            for _ in range(num_blocks)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        # Use the same random seed as the official DimeNet` implementation.
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, z, pos, batch=None):
        """"""
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
