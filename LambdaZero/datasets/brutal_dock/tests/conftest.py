"""
This file defines pytest fixtures. These are then auto-discovered by pytest
when the tests are executed.
"""
import pytest
import torch
from torch_geometric.data import Data, Batch


def get_list_edge_indices_for_a_ring(number_of_nodes):
    list_edges_indices = [[number_of_nodes - 1, 0], [0, number_of_nodes - 1]]
    for node_index in range(number_of_nodes - 1):
        list_edges_indices.append([node_index, node_index + 1])
        list_edges_indices.append([node_index + 1, node_index])

    return list_edges_indices


def test_get_list_edge_indices_for_a_ring():
    number_of_nodes = 3
    expected_list = [[2, 0], [0, 2], [0, 1], [1, 0], [1, 2], [2, 1]]
    computed_list = get_list_edge_indices_for_a_ring(number_of_nodes)

    assert expected_list == computed_list


@pytest.fixture
def number_of_nodes():
    """
    How many atoms in the fake molecule.
    """
    return 6


@pytest.fixture
def positions(number_of_nodes):
    """
    Random positions for the atoms
    """
    torch.random.manual_seed(34534)
    pos = torch.rand(number_of_nodes, 3)

    return pos


@pytest.fixture
def dockscore():
    torch.random.manual_seed(3245)
    return torch.rand(1)


@pytest.fixture
def number_of_node_features():
    """
    How many features are on a graph node. Free parameter.
    """
    return 17


@pytest.fixture
def number_of_edge_features():
    """
    How many features are on a graph node.
    Seems to be hardcoded to 4 in the MessagePassingNet model.
    """
    return 4


@pytest.fixture
def random_molecule_data(number_of_nodes, number_of_node_features, positions,
                         number_of_edge_features, dockscore):
    """
    Simple "molecule" in torch geometric. The molecule will be a simple atomic ring.

    From inspection, a molecule with 44 atoms and 50 bonds has data of the form
     >> Data(dockscore=[1], edge_attr=[100, 4], edge_index=[2, 100], pos=[44, 3], x=[44, 14])
    """
    torch.random.manual_seed(123)

    # a simple ring, with directional edges
    number_of_edges = 2*number_of_nodes

    edge_attr = torch.rand(number_of_edges, number_of_edge_features)

    # edges are connecting adjacent nodes, with a periodic condition where the last
    # is connected to the zeroth node.
    list_edges_indices = get_list_edge_indices_for_a_ring(number_of_nodes)
    edge_index = torch.tensor(list_edges_indices).transpose(1, 0)

    node_data = torch.rand(number_of_nodes, number_of_node_features)

    fake_molecule_data = Data(dockscore=dockscore,
                              edge_attr=edge_attr,
                              edge_index=edge_index,
                              pos=positions,
                              x=node_data)

    return fake_molecule_data


@pytest.fixture
def random_molecule_batch(random_molecule_data):
    return Batch.from_data_list([random_molecule_data])
