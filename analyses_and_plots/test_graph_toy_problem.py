from collections import OrderedDict

import pytest
import torch
from torch_geometric.data import Data

from analyses_and_plots.graph_toy_problem import make_graph_from_dict, MAX_NUMBER_OF_NODES, MAX_NUMBER_OF_EDGES, \
    make_dict_from_graph


@pytest.fixture
def x():
    return torch.tensor([[1], [1], [1]], requires_grad=False)


@pytest.fixture
def edge_index():
    return torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long, requires_grad=False)


@pytest.fixture
def graph(x, edge_index):
    graph = Data(x=x, edge_index=edge_index)
    return graph


@pytest.fixture
def data_dict(x, edge_index):
    zero_padded_x = torch.zeros(MAX_NUMBER_OF_NODES, 1, dtype=torch.int)
    zero_padded_edge_index = torch.zeros(2, MAX_NUMBER_OF_EDGES, dtype=torch.long)

    zero_padded_x[:len(x), :] = x
    zero_padded_edge_index[:, :edge_index.shape[1]] = edge_index

    cutoff = (len(x), edge_index.shape[1])

    return OrderedDict({'cutoff': cutoff, 'x': zero_padded_x, 'edge_index': zero_padded_edge_index})


def test_make_graph_from_dict(graph, data_dict):
    computed_graph = make_graph_from_dict(data_dict)

    assert torch.all(torch.eq(graph.x, computed_graph.x))
    assert torch.all(torch.eq(graph.edge_index, computed_graph.edge_index))


def test_make_dict_from_graph(graph, data_dict):
    computed_data_dict = make_dict_from_graph(graph)

    assert computed_data_dict['cutoff'] == data_dict['cutoff']
    assert torch.all(torch.eq(computed_data_dict['x'], data_dict['x']))
    assert torch.all(torch.eq(computed_data_dict['edge_index'], data_dict['edge_index']))
