"""
This file defines pytest fixtures. These are then auto-discovered by pytest
when the tests are executed.
"""
import logging
import tempfile

import pytest
import torch
from torch_geometric.data import Batch

from LambdaZero.datasets.brutal_dock.tests.fake_molecules import get_list_edge_indices_for_a_ring, \
    get_random_molecule_data


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
    torch.random.manual_seed(123)
    fake_molecule_data = get_random_molecule_data(number_of_nodes, number_of_node_features,
                                                  positions, number_of_edge_features, dockscore)
    return fake_molecule_data


@pytest.fixture
def random_molecule_batch(random_molecule_data):
    return Batch.from_data_list([random_molecule_data])


@pytest.fixture
def tracking_uri():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield tmp_dir_str
    logging.info("deleting test folder")


@pytest.fixture
def experiment_name():
    return 'some-fake-experiment-name'