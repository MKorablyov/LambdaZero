"""
This file defines pytest fixtures. These are then auto-discovered by pytest
when the tests are executed.
"""
import logging
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.dataset_utils import get_molecule_graphs_from_raw_data_dataframe
from LambdaZero.datasets.brutal_dock.models.message_passing_model import MessagePassingNet
from LambdaZero.datasets.brutal_dock.tests.fake_molecules import get_random_molecule_data
from LambdaZero.datasets.brutal_dock.tests.fake_molecule_dataset import FakeMoleculeDataset
from LambdaZero.datasets.brutal_dock.tests.testing_utils import generate_random_string


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
def realistic_hgraph_vocab():
    return {
        ('NN', 'N[NH2:1]'),
        ('CN', 'N[CH3:1]'),
        ('C1=CNCN=C1', 'C1=C[NH:1]CN=C1'),
        ('C1CCOCC1', 'C1COC[CH2:1]C1'),
        ('CN', 'C[NH2:1]'),
        ('C1=CN=CNC1', 'C1=NC=[CH:1]CN1'),
        ('C1=CC=CC=C1', 'C1=CC=[CH:1]C=C1'),
        ('SS', 'S[SH:1]'),
        ('C[NH3+]', '[NH3+][CH3:1]'),
        ('C1=CC=[NH+]C=C1', 'C1=CC=[NH+:1]C=C1'),
        ('C1=CNCNC1', 'C1=CN[CH2:1]NC1'),
        ('C1=NCCN1', 'C1=NCC[NH:1]1'),
        ('C1=CSC=N1', 'C1=N[CH:1]=CS1'),
        ('C=O', 'C=[O:1]'),
        ('C1COCC[NH2+]1', 'C1C[NH2+:1]CCO1'),
        ('C#N', 'N#[CH:1]'),
        ('C1COCCN1', 'C1C[NH:1]CCO1'),
        ('CP', 'P[CH3:1]'),
        ('C1=CNCNC1', 'C1=[CH:1]CNCN1'),
        ('CF', 'F[CH3:1]'),
        ('O=P', 'O=[PH:1]'),
        ('OS', 'O[SH:1]'),
        ('NP', 'P[NH2:1]'),
        ('C=O', 'O=[CH2:1]'),
        ('C1CCNC1', 'C1C[CH2:1]CN1'),
        ('C=N', 'C=[NH:1]'),
        ('C1CNCCN1', 'C1C[NH:1]CCN1'),
        ('C1=CN=CNC1', 'C1=N[CH:1]=CCN1'),
        ('[O-]P', '[O-][PH2:1]'),
        ('NI', 'I[NH2:1]'),
        ('CBr', 'Br[CH3:1]'),
        ('C1=CNCCN1', 'C1=CN[CH2:1]CN1'),
        ('C1=C[NH]C=C1', 'C1=C[CH:1]=C[NH]1'),
        ('C[SH2+]', 'C[SH2+:1]'),
        ('S', 'S'),
        ('C1=CCCC=C1', 'C1=C[CH2:1][CH2:1]C=C1'),
        ('C1=CCCC=C1', 'C1=CC[CH2:1]C=C1'),
        ('NS', 'S[NH2:1]'),
        ('C1=C[NH]C=C1', 'C1=C[NH][CH:1]=C1'),
        ('C=C', 'C=[CH2:1]'),
        ('C1=CN=CN=C1', 'C1=NC=[CH:1]C=N1'),
        ('OP', 'O[PH2:1]'),
        ('CS', 'S[CH3:1]'),
        ('O=S', 'O=[S:1]'),
        ('C1=CN=CN=C1', 'C1=C[CH:1]=NC=N1'),
        ('C1=CCCCC1', 'C1=[CH:1]CCCC1'),
        ('C1=CN=CNC1', 'C1=N[CH:1]=[CH:1]CN1'),
        ('O=S', 'S=[O:1]'),
        ('C1=CN=CN=C1', 'C1=NC=[CH:1][CH:1]=N1'),
        ('P', 'P'),
        ('C', 'C'),
        ('CC', 'C[CH3:1]'),
        ('C1=CNCN=C1', 'C1=C[CH:1]=NCN1'),
        ('C1=CNC=CC1', 'C1=CNC=[CH:1]C1'),
        ('C[SH2+]', '[SH2+][CH3:1]'),
        ('C1=CNC=CC1', 'C1=C[NH:1]C=CC1'),
        ('CO', 'O[CH3:1]'),
        ('CCl', 'Cl[CH3:1]')
    }


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


@pytest.fixture
def fake_raw_molecule_data_dataframe(realistic_smiles):
    np.random.seed(12312)
    size = len(realistic_smiles)
    list_gridscores = np.random.random(size)
    list_klabels = np.random.choice(range(10), size)

    df = pd.DataFrame(data={'smiles': realistic_smiles, 'gridscore': list_gridscores, 'klabel': list_klabels})

    return df


@pytest.fixture
def original_raw_data_dir(fake_raw_molecule_data_dataframe):
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        raw_dir = Path(tmp_dir_str).joinpath("raw")
        raw_dir.mkdir()

        number_of_files = len(D4MoleculesDataset.feather_filenames)
        list_index_groups = np.array_split(fake_raw_molecule_data_dataframe.index, number_of_files)

        for filename, indices in zip(D4MoleculesDataset.feather_filenames, list_index_groups):
            file_path = raw_dir.joinpath(filename)
            sub_df = fake_raw_molecule_data_dataframe.iloc[indices].reset_index(drop=True)
            sub_df.to_feather(file_path)
        yield str(raw_dir)

    logging.info("deleting test folder")


@pytest.fixture
def root_dir():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        yield tmp_dir_str

    logging.info("deleting test folder")
