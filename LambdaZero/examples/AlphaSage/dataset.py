import torch
import numpy as np
from rdkit import Chem
import networkx as nx

import LambdaZero.utils
import LambdaZero.environments
import LambdaZero.inputs
from copy import deepcopy

class MolMaxDist:
    def __init__(self, steps, blocks_file):
        self.steps = steps
        self.molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=blocks_file)

    def __call__(self):
        # todo: paralelize this call 10/40 sec per iteration depending on it
        self.molMDP.reset()
        self.molMDP.random_walk(self.steps)
        # print(self.molMDP.molecule.numblocks)
        # print(Chem.MolToSmiles(self.molMDP.molecule.mol))
        graph = mol_to_graph(self.molMDP.molecule.mol)
        graph = precompute_edge_slices(graph)
        graph = precompute_max_dist(graph)
        return graph

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
    # print((slices0 + slices1)[-1], graph.edge_attr.shape)
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
