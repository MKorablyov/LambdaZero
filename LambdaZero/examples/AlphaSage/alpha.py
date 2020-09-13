import time, os.path as osp
import torch
import numpy as np
from torch_sparse import SparseTensor
import torch_scatter
#from torch_scatter import scatter, scatter_max, scatter_sum, scatter_logsumexp
from rdkit import Chem
import networkx as nx

import LambdaZero.utils
import LambdaZero.environments
import LambdaZero.inputs
from copy import deepcopy

from mlp import MLP, GRUAggr
from ray import tune

def slice_cat(tensor,slices, dim):
    chunks = [tensor.narrow(dim, slice[0], slice[1]) for slice in slices]
    return torch.cat(chunks,dim=dim)

# todo: torch jit optimize
def repeat_cat(tensor, repeats):
    _once = np.ones(len(tensor.shape),dtype=np.int)[1:]
    chunks = [tensor.narrow(0,i,1).repeat(repeat, *_once) for i,repeat in enumerate(repeats)]
    return torch.cat(chunks,dim=0)


class DiffGCN(torch.nn.Module):
    def __init__(self, channels, t, eps):
        super(DiffGCN, self).__init__()
        self.channels = channels
        self.t = t
        self.eps = eps
        self.diff_mlp = MLP([channels[0]*(1+t), 64, 1])
        #self.aggr_mlp = MLP([channels[0]*(1+t), 64, channels[1]])
        self.aggr = GRUAggr(channels[0], channels[1], 128, 1)

    def _walk_logp(self, walk_embeds_t):
        "copute probability of each walk on the graph"
        walk_embeds_t = walk_embeds_t.view([walk_embeds_t.shape[0], self.channels[0] * (1+self.t)])
        # unscaled walk probs
        with torch.no_grad():  # diffusion gradient provided separately
            walk_logp_t = self.diff_mlp(walk_embeds_t)[:, 0]
        return walk_logp_t

    def diffuse(self, v, adj, slices):
        num_nodes, d = v.shape
        # initialize
        walks = torch.arange(num_nodes,device=v.device)[:,None] # [w,t]
        walk_embeds = torch.zeros([num_nodes, 1+self.t, self.channels[0]], dtype=torch.float, device=v.device)#[w, d*t]
        walk_embeds[:,0,:] = v

        # diffuse
        for t in range(self.t):
            # find slices of the adjacency matrix
            slices_t = slices[walks[:, -1]]
            # make new adjacency matrix;
            adj_t = slice_cat(adj, slices_t, dim=1)
            # get node_in embeddings
            v_t = v[adj_t[1]]
            # form timestep walk embeds
            walk_embeds_t = repeat_cat(walk_embeds, slices_t[:,1])
            walk_embeds_t[:, 1+t, :] = v_t

            walk_logp_t = self._walk_logp(walk_embeds_t)
            # rescale walk probs
            init_vs = repeat_cat(walks[:,0], slices_t[:,1])
            norm = torch_scatter.scatter_logsumexp(walk_logp_t, init_vs, dim=0, dim_size=num_nodes)
            norm = repeat_cat(norm,slices_t[:,1])
            walk_p_t = torch.exp(walk_logp_t - norm)
            # print(scatter(walk_p_t,init_vs, reduce="sum", dim=0, dim_size=num_nodes))
            # add_noise
            # todo: categorical noise and why
            noise = self.eps * torch.randn(walk_p_t.shape[0],device=v.device)
            walk_p_t = walk_p_t + noise
            _, walks_t = torch_scatter.scatter_max(walk_p_t, init_vs, dim=0, dim_size=num_nodes)
            walks_t = adj_t[1][walks_t]
            # update walks and walk embeddings
            walks = torch.cat([walks, walks_t[:,None]], dim=1)
            walk_embeds[:,1+t, :] = v[walks_t,:]
        return walks, walk_embeds

    def aggregate(self, walks, walk_embeds): # [e,d] -> [v,d]
        #num_nodes = walk_embeds.shape[0]
        #walk_embeds = walk_embeds.view([num_nodes, self.channels[0] * (1+self.t)])
        return self.aggr(walk_embeds)

    def forward(self, node_attr, edge_index, slices):
        walks, walk_embeds = self.diffuse(node_attr, edge_index, slices)
        # print(walk_embeds.sum(dim=[2]))
        v_out = self.aggregate(walks, walk_embeds)
        return v_out

    # def backward_hook()
        # rewards = F([s0,s1,s2,s3], l2_loss)
        # R * log (prob(a | s))


class SimpleGCN(torch.nn.Module):
    def __init__(self, channels):
        super(SimpleGCN, self).__init__()
        self.mlp = MLP(channels)

    def message(self, x, adj): # [v,d] -> [e,d]
        edge_x = x[adj[0]]
        return edge_x

    def aggregate(self, edge_x, adj, num_nodes): # [e,d] -> [v,d]
        x = torch_scatter.scatter(edge_x, adj[1], reduce="max", dim=0, dim_size=num_nodes)
        return x

    def combine(self, x1, x2):
        return self.mlp(x1 + x2)

    def forward(self, x1, adj):
        num_nodes = x1.shape[0]
        edge_x = self.message(x1, adj)
        x2 = self.aggregate(edge_x, adj, num_nodes)
        x3 = self.combine(x1,x2)
        return x3

def mol_to_graph(mol):
    mol = Chem.RemoveHs(mol)
    atmfeat, coord, bond, bondfeat = LambdaZero.inputs.mpnn_feat(mol, ifcoord=False)
    graph = LambdaZero.inputs._mol_to_graph(atmfeat, coord, bond, bondfeat, {})
    return graph

def precompute_edge_slices(graph):
    # add inverse direction
    edges = graph.edge_index.numpy()
    assert np.all(np.diff(edges[0]) >= 0), "implement me: edge array was not sorted"
    _, slices1 = np.unique(edges[0], return_counts=True)
    slices0 = np.concatenate([np.array([0]), np.cumsum(slices1)])[:-1]
    #print((slices0 + slices1)[-1], graph.edge_attr.shape)
    graph.slices = torch.tensor(np.stack([slices0, slices1], axis=1))
    return graph

def precompute_max_dist(graph):
    # compute distance to the furthers atom for each
    e = graph.edge_index.numpy()
    G = nx.Graph()
    G.add_edges_from(e.T)
    p = nx.shortest_path_length(G)
    graph.max_dist = torch.tensor([max(l[1].values()) for l in p])
    return graph

class MolMaxDist:
    def __init__(self, steps, blocks_file):
        self.steps = steps
        self.molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=blocks_file)

    def __call__(self):
        self.molMDP.reset()
        self.molMDP.random_walk(self.steps)
        graph = mol_to_graph(self.molMDP.molecule.mol)
        graph = precompute_edge_slices(graph)
        graph = precompute_max_dist(graph)
        return graph


class AlphaSageTrainer(tune.Trainable):
    def _setup(self, config):
        self.config = config
        self.dataset = MolMaxDist(config["dataset_mdp_steps"],config["blocks_file"])
        self.conv = config["model"]([14, 1], eps=config["eps"], t=config["t"])
        self.optim = config["optimizer"](self.conv.parameters(), **config["optimizer_config"])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv.to(self.device)

    def _train(self):
        losses = []
        for i in range(self.config["dataset_size"]):
            self.optim.zero_grad()
            # for i in range(self.config["b_size"]):
            g = self.dataset()
            g = g.to(self.device)

            x = self.conv(g.x, g.edge_index, g.slices)
            norm_dist = (g.max_dist - 10) / 8.5
            #print("x", x[:,0].shape, "norm_dist", norm_dist.shape)
            loss = ((x[:,0] - norm_dist) ** 2).mean()
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            self.optim.step()
        loss_mean = np.mean(losses)
        return {"loss_mean":loss_mean}


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
DEFAULT_CONFIG = {
    "regressor_config": {
        "run_or_experiment": AlphaSageTrainer,
        "config": {
            "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
            "dataset_mdp_steps": 1,
            "b_size": 50,
            "model": DiffGCN,
            "optimizer": torch.optim.Adam,
            "optimizer_config": {"lr": 0.001,},
            "eps": 100,
            "t": 7,
            "dataset_size":1000,
            "device":"cuda",
        },
        "local_dir": summaries_dir,
        "stop": {"training_iteration": 1000
                 },
        "resources_per_trial": {
            "cpu": 4,
            "gpu": 1.0
        },
        "checkpoint_score_attr":"train_loss",
        "num_samples": 1,
        "checkpoint_at_end": False,
    },
    "memory": 10 * 10 ** 9,
}


# todo: my current understanding is that there is no bug, but features are incredibly weak

if __name__ == "__main__":
    config = DEFAULT_CONFIG
    #trainer = AlphaSageTrainer(config["trainer_config"])
    #metrics = trainer._train()
    #print(metrics)

    tune.run(**config["regressor_config"])

    # v1 = torch.arange(12,dtype=torch.float).reshape(3,4)
    # adj = torch.tensor([[0,0,1,2],
    #                     [1,2,0,0]])
    # slices = torch.tensor([[0,2],
    #                        [2,1],
    #                        [3,1]])