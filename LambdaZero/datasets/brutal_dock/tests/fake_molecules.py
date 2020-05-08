import torch
from torch_geometric.data import Data
import numpy as np


def get_list_edge_indices_for_a_ring(number_of_nodes):
    list_edges_indices = [[number_of_nodes - 1, 0], [0, number_of_nodes - 1]]
    for node_index in range(number_of_nodes - 1):
        list_edges_indices.append([node_index, node_index + 1])
        list_edges_indices.append([node_index + 1, node_index])

    return list_edges_indices


def get_random_molecule_data(number_of_nodes, number_of_node_features, positions,
                             number_of_edge_features, gridscore, klabel):
    """
    Simple "molecule" in torch geometric. The molecule will be a simple atomic ring.

    From inspection, a molecule with 44 atoms and 50 bonds has data of the form
     >> Data(gridscore=[1], edge_attr=[100, 4], edge_index=[2, 100], pos=[44, 3], x=[44, 14])
    """
    # a simple ring, with directional edges
    number_of_edges = 2*number_of_nodes

    edge_attr = torch.rand(number_of_edges, number_of_edge_features)

    # edges are connecting adjacent nodes, with a periodic condition where the last
    # is connected to the zeroth node.
    list_edges_indices = get_list_edge_indices_for_a_ring(number_of_nodes)
    edge_index = torch.tensor(list_edges_indices).transpose(1, 0)

    node_data = torch.rand(number_of_nodes, number_of_node_features)

    fake_molecule_data = Data(gridscore=gridscore,
                              klabel=klabel,
                              edge_attr=edge_attr,
                              edge_index=edge_index,
                              pos=positions,
                              x=node_data,
                              test_tag=np.random.randint(1, 1e9))

    return fake_molecule_data
