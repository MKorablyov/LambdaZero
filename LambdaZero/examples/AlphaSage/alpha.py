import time, os.path as osp
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_max, scatter_sum, scatter_logsumexp
from rdkit import Chem
import networkx as nx

import LambdaZero.utils
import LambdaZero.environments
import LambdaZero.inputs
from copy import deepcopy

from mlp import MLP


def slice_cat(tensor,slices, dim):
    chunks = [tensor.narrow(dim, slice[0], slice[1]) for slice in slices]
    return torch.cat(chunks,dim=dim)

def repeat_cat(tensor, repeats):
    _once = np.ones(len(tensor.shape),dtype=np.int)[1:]
    chunks = [tensor.narrow(0,i,1).repeat(repeat, *_once) for i,repeat in enumerate(repeats)]
    return torch.cat(chunks,dim=0)


class DiffGCN(torch.nn.Module):
    def __init__(self, channels, t, eps):
        super(DiffGCN, self).__init__()
        self.t = t
        self.eps = eps
        self.diff_mlp = MLP([channels[0]*t, 1])
        self.aggr_mlp = MLP([channels[0]*t, channels[1]])

    def diffuse(self, v, adj, slices):
        num_nodes, d = v.shape

        # initialize
        walks = torch.arange(num_nodes)[:,None] # [w,t]
        walk_embeds = torch.zeros(num_nodes, d*self.t, dtype=torch.float) # [w, d*t]

        # diffuse
        for i in range(self.t):
            # find slices of the adjacency matrix
            slices_t = slices[walks[:, -1]]
            # make new adjacency matrix;
            adj_t = slice_cat(adj, slices_t,dim=1)
            # get node_in embeddings
            v_t = v[adj_t[1]]
            # form timestep walk embeds
            walk_embeds_t = repeat_cat(walk_embeds, slices_t[:,1])
            walk_embeds_t[:, i*d:(i+1)*d] += v_t
            # unscaled walk probs

            with torch.no_grad(): # diffusion gradient provided separately
                walk_logp_t = self.diff_mlp(walk_embeds_t)[:,0]

            # rescale walk probs
            init_vs = repeat_cat(walks[:,0], slices_t[:,1])
            norm = scatter_logsumexp(walk_logp_t, init_vs, dim=0, dim_size=num_nodes)
            norm = repeat_cat(norm,slices_t[:,1])
            walk_p_t = torch.exp(walk_logp_t - norm)
            # print(scatter(walk_p_t,init_vs, reduce="sum", dim=0, dim_size=num_nodes))
            # add_noise
            # todo: categorical noise and why
            noise = self.eps * torch.randn(walk_p_t.shape[0])
            walk_p_t = walk_p_t + noise
            _, walks_t = scatter_max(walk_p_t, init_vs, dim=0, dim_size=num_nodes)
            walks_t = adj_t[1][walks_t]
            # update walks and walk embeddings
            walks = torch.cat([walks, walks_t[:,None]],dim=1)
            walk_embeds[:, i*d:(i+1)*d] += v[walks_t,:]
        return walks, walk_embeds

    def aggregate(self, walks, walk_embeds): # [e,d] -> [v,d]
        return self.aggr_mlp(walk_embeds)

    def forward(self, node_attr, edge_index, slices):
        #v, adj, slices = graph.x, graph.edge_index, graph.slices
        walks, walk_embeds = self.diffuse(node_attr, edge_index, slices)
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
        x = scatter(edge_x, adj[1], reduce="max", dim=0, dim_size=num_nodes)
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
    def __init__(self, blocks_file):
        self.molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=blocks_file)

    def __call__(self):
        self.molMDP.reset()
        self.molMDP.random_walk(5)
        graph = mol_to_graph(self.molMDP.molecule.mol)
        graph = precompute_edge_slices(graph)
        graph = precompute_max_dist(graph)
        return graph




if __name__ == "__main__":
    v1 = torch.arange(12,dtype=torch.float).reshape(3,4)
    adj = torch.tensor([[0,0,1,2],
                        [1,2,0,0]])
    slices = torch.tensor([[0,2],
                           [2,1],
                           [3,1]])

    conv = DiffGCN([14, 4],eps=0.2,t=3)
    #conv(v1,adj, slices)

    datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
    dataset = MolMaxDist(osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"))

    # todo: create a benchmark for molecule graph; estimate largest distance for every atom L2 loss
    # todo: normalize distances

    for i in range(1000):
        g = dataset()
        print(g)
        conv(g.x, g.edge_index, g.slices)
