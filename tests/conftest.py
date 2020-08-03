"""
This file defines pytest fixtures. These are then auto-discovered by pytest
when the tests are executed.
"""
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import pytest
import torch
from torch_geometric.data import Batch

from LambdaZero.datasets.temp_brunos_work.dataset_utils import get_molecule_graphs_from_raw_data_dataframe
from LambdaZero.utils import get_external_dirs
from tests.fake_molecule_dataset import FakeMoleculeDataset
from tests.fake_molecules import get_random_molecule_data
from tests.testing_utils import generate_random_string


def pytest_addoption(parser):
    parser.addoption(
        "--external_program", action="store_true", default=False, help="run external program integration tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--external_program"):
        # if --external_program is given in cli, then do not skip external program integration smoke tests
        return
    skip_external_program_integration = pytest.mark.skip(reason="need --external_program option to run")
    for item in items:
        if "external_program" in item.keywords:
            item.add_marker(skip_external_program_integration)


def pytest_configure(config):
    """
    The code base peppers calls to get_external_dirs() all over the place. This breaks the tests
    if the installation script "install-prog-data.sh" has not been executed to create the
    external_dirs.cfg file. This operation is expensive and time consuming: it's fine on a local
    machine, but we don't want to do it all the time on the CI server. Here we'll create a fake
    config file if it doesn't exist.

    """

    try:
        get_external_dirs()
    except ImportError:
        logging.warning("The external directories are not configured. Creating a fake config file for testing.")

        config_file_path = Path(__file__).parent.parent.joinpath("external_dirs.cfg")

        with open(config_file_path, 'w') as f:
            f.write('[dir]\n')
            f.write('datasets=/some/fake/path/datasets/\n')
            f.write('programs=/some/fake/path/programs/\n')
            f.write('summaries=/some/fake/path/summaries/')


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
    list_smiles = ['CC(C)=CC(C)(C)O',
                   'CC(C)=CC(=O)NC(C#N)P(=O)(O)O',
                   'O=[SH](=O)S(=O)(=O)O',
                   'CC(C)=CN1CCN(P(=O)(O)O)CC1',
                   'CC(C)(O)C(F)(F)F',
                   'c1ccc2cc(N3CCOCC3)ccc2c1',
                   'CC(C)(O)Br',
                   'CC(=O)N[SH](=O)=O',
                   'CC(C)=CC1CC(C(NC(=O)C(C)O)NC(=O)S(=O)(=O)O)N(c2ccc3ccccc3c2)C1C(C)C',
                   'C1=C(c2ccc[nH]2)CCCC1',
                   'O=C(NCF)C1=CCCCC1',
                   'CC(=Cc1cc[nH]c1)CCCl',
                   'CC(=O)NC(=O)NC1=CN(I)C=CC1']

    return list_smiles


@pytest.fixture
def hard_smiles():
    list_smiles = ['Nn1cnc2c([PH](=O)[O-])ncnc21',
                   'N=C[n+]1cccc(-[n+]2cccc(C(NC(=O)c3csc(N4C=CCC(S)=C4)n3)[NH+]3CCOCC3)c2)c1',
                   'O=C(NC([NH+]1CCOCC1)[PH](=O)O)n1ccc(S(=O)(=O)O)nc1=O',
                   'CC(O)c1cn(C(C)O)c(=O)[nH]c1=O',
                   'CC(=O)NCc1ccn(C2CNc3nc(-c4ccc[nH]4)[nH]c(=O)c3N2)c(=O)n1',
                   'C[SH+]c1cc[nH]c1',
                   'O=c1[nH]cc(-n2cnc3c(C4C(C5CNc6nc(C(F)(F)F)[nH]c(=O)c6N5)CCOC4[n+]4ccccc4)ncnc32)c(=O)[nH]1',
                   ]
    return list_smiles


@pytest.fixture
def realistic_smiles(easy_smiles, hard_smiles):
    list_smiles = []
    list_smiles.extend(easy_smiles)
    list_smiles.extend(hard_smiles)
    return list_smiles


@pytest.fixture
def smiles_and_scores_dataframe(realistic_smiles):

    np.random.seed(213421)
    number_of_smiles = len(realistic_smiles)
    klabels = np.random.choice(np.arange(20), number_of_smiles)
    gridscores = np.random.rand(number_of_smiles)
    mol_names = [generate_random_string(15) for _ in range(number_of_smiles)]

    df = pd.DataFrame(data={'smiles': realistic_smiles,
                            'mol_name': mol_names,
                            'gridscore': gridscores,
                            'klabel': klabels})

    return df


@pytest.fixture
def list_real_molecules(smiles_and_scores_dataframe):
    list_graphs = get_molecule_graphs_from_raw_data_dataframe(smiles_and_scores_dataframe)
    return list_graphs


@pytest.fixture
def real_molecule_batch(list_real_molecules):
    return Batch.from_data_list(list_real_molecules)


@pytest.fixture
def real_molecule_dataset(list_real_molecules):
    return FakeMoleculeDataset(list_real_molecules)
