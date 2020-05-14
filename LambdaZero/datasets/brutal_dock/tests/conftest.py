"""
This file defines pytest fixtures. These are then auto-discovered by pytest
when the tests are executed.
"""
import logging
import tempfile
import numpy as np

import pytest
import torch
from torch_geometric.data import Batch

from LambdaZero.datasets.brutal_dock.models.message_passing_model import MessagePassingNet
from LambdaZero.datasets.brutal_dock.tests.fake_molecules import get_random_molecule_data
from LambdaZero.datasets.brutal_dock.tests.fake_molecule_dataset import FakeMoleculeDataset


@pytest.fixture
def number_of_molecules():
    return 1


@pytest.fixture
def list_of_node_count(number_of_molecules):
    """
    How many atoms in the fake molecules.
    """
    return list(range(3, 3 + number_of_molecules))


@pytest.fixture
def list_positions(list_of_node_count):
    """
    Random positions for the atoms
    """
    torch.manual_seed(12312)
    list_positions = []
    for number_of_nodes in list_of_node_count:
        pos = torch.rand(number_of_nodes, 3, requires_grad=False)
        list_positions.append(pos)

    return list_positions


@pytest.fixture
def list_gridscores(number_of_molecules):
    torch.manual_seed(242)
    return torch.rand((number_of_molecules, 1), requires_grad=False)


@pytest.fixture
def list_klabels(number_of_molecules):
    np.random.seed(213423)
    klabels = np.random.choice(list(range(10)), number_of_molecules)
    return torch.from_numpy(klabels).view(-1, 1)


@pytest.fixture
def number_of_node_features():
    """
    How many features are on a graph node. Free parameter.
    """
    return 3


@pytest.fixture
def number_of_edge_features():
    """
    How many features are on a graph node.
    Seems to be hardcoded to 4 in the MessagePassingNet model.
    """
    return 4


@pytest.fixture
def list_random_molecules(list_of_node_count, number_of_node_features, list_positions,
                          number_of_edge_features, list_gridscores, list_klabels):

    list_molecules = []
    for number_of_nodes, positions, gridscore, klabel in \
            zip(list_of_node_count, list_positions, list_gridscores, list_klabels):
        fake_molecule_data = get_random_molecule_data(number_of_nodes,
                                                      number_of_node_features,
                                                      positions,
                                                      number_of_edge_features,
                                                      gridscore,
                                                      klabel)
        list_molecules.append(fake_molecule_data)
    return list_molecules


@pytest.fixture
def random_molecule_data(list_random_molecules):
    """
    Return a single molecule.
    """
    return list_random_molecules[0]


@pytest.fixture
def random_molecule_batch(list_random_molecules):
    return Batch.from_data_list(list_random_molecules)


@pytest.fixture
def random_molecule_dataset(list_random_molecules):
    return FakeMoleculeDataset(list_random_molecules)


@pytest.fixture
def mpnn_model(number_of_node_features):

    model_instantiation_parameters = dict(node_feat=number_of_node_features,
                                          edge_feat=4, gcn_size=8, edge_hidden=8,
                                          gru_layers=1, linear_hidden=8, out_size=1)

    mpnn = MessagePassingNet.create_model_for_training(model_instantiation_parameters)

    return mpnn


@pytest.fixture
def tracking_uri():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield tmp_dir_str
    logging.info("deleting test folder")


@pytest.fixture
def experiment_name():
    return 'some-fake-experiment-name'


@pytest.fixture
def easy_smiles():
    list_smiles = ['CC(C)=CC(C)(C)O', 'CC(C)=CC(=O)NC(C#N)P(=O)(O)O',
                   'O=[SH](=O)S(=O)(=O)O', 'CC(C)=CN1CCN(P(=O)(O)O)CC1',
                   'CC(C)(O)C(F)(F)F', 'c1ccc2cc(N3CCOCC3)ccc2c1', 'CC(C)(O)Br',
                   'CC(=O)N[SH](=O)=O', 'CC(C)=CC1CC(C(NC(=O)C(C)O)NC(=O)S(=O)(=O)O)N(c2ccc3ccccc3c2)C1C(C)C']
    return list_smiles


@pytest.fixture
def hard_smiles():
    list_smiles = ['Nn1cnc2c([PH](=O)[O-])ncnc21']
    return list_smiles


@pytest.fixture
def realistic_smiles(easy_smiles, hard_smiles):
    list_smiles = []
    list_smiles.extend(easy_smiles)
    list_smiles.extend(hard_smiles)
    return list_smiles
