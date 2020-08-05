import json
import logging
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray
from rdkit.Chem.rdmolfiles import MolFromSmiles
from torch_geometric.data import DataLoader

from LambdaZero.examples.env3d.dataset.processing import env3d_proc
from LambdaZero.inputs import BrutalDock


@pytest.fixture
def debug_blocks():
    """
    A subset of the blocks_PDB_105.json file for testing purposes.
    """
    debug_blocks = {
        "block_name": {
            "0": "c1ccccc1_0",
            "1": "CO_0",
            "2": "CO_1",
            "3": "C=O_0",
            "4": "C1CCNCC1_0",
            "5": "C1CCNCC1_1",
            "6": "C1CCNCC1_5",
            "7": "O_0",
            "8": "C_0",
            "9": "c1ccncc1_0",
        },
        "block_smi": {
            "0": "c1ccccc1",
            "1": "CO",
            "2": "CO",
            "3": "C=O",
            "4": "C1CCNCC1",
            "5": "C1CCNCC1",
            "6": "C1CCNCC1",
            "7": "O",
            "8": "C",
            "9": "c1ccncc1",
        },
        "block_r": {
            "0": [0, 1, 2, 3, 4, 5],
            "1": [0, 1],
            "2": [1, 0, 0, 0],
            "3": [0],
            "4": [0, 3],
            "5": [1, 0, 3],
            "6": [3, 0],
            "7": [0],
            "8": [0],
            "9": [0, 1, 2, 4, 5],
        },
    }
    return debug_blocks


@pytest.fixture
def blocks_file(debug_blocks):
    with tempfile.TemporaryDirectory() as tmp_dir:

        blocks_file = Path(tmp_dir).joinpath("blocks_debug.json")
        with open(blocks_file, "w") as f:
            json.dump(debug_blocks, f)

        yield str(blocks_file)


@pytest.fixture
def smiles():
    list_smiles = [
        "CC(C)=CC(C)(C)O",
        "CC(C)=CC(=O)NC(C#N)P(=O)(O)O",
        "O=[SH](=O)S(=O)(=O)O",
        "CC(C)=CN1CCN(P(=O)(O)O)CC1",
        "CC(C)(O)C(F)(F)F",
        "c1ccc2cc(N3CCOCC3)ccc2c1",
        "CC(C)(O)Br",
        "CC(=O)N[SH](=O)=O",
        "CC(C)=CC1CC(C(NC(=O)C(C)O)NC(=O)S(=O)(=O)O)N(c2ccc3ccccc3c2)C1C(C)C",
        "C1=C(c2ccc[nH]2)CCCC1",
        "O=C(NCF)C1=CCCCC1",
        "CC(=Cc1cc[nH]c1)CCCl",
        "CC(=O)NC(=O)NC1=CN(I)C=CC1",
    ]
    return list_smiles


@pytest.fixture
def data_df(smiles):
    np.random.seed(231)

    list_rows = []

    for s in smiles:

        mol = MolFromSmiles(s)
        number_of_atoms = mol.GetNumAtoms()
        coord = np.random.rand(number_of_atoms, 3)

        n_axis = np.random.rand(3)
        n_axis /= np.linalg.norm(n_axis)

        theta = 2 * np.pi * np.random.rand()
        block_index = np.random.randint(0, 105)
        attachment_node_index = np.random.randint(0, number_of_atoms)

        row = {
            "smi": s,
            "coord": coord,
            "n_axis": n_axis,
            "attachment_node_index": attachment_node_index,
            "attachment_angle": theta,
            "attachment_block_index": block_index,
        }

        list_rows.append(row)

    df = pd.DataFrame(data=list_rows)

    return df


@pytest.fixture
def dataset_root_and_filename(data_df):
    filename = "test_data"

    df_for_writing = copy(data_df)
    df_for_writing["coord"] = df_for_writing["coord"].apply(np.ndarray.tobytes)
    df_for_writing["n_axis"] = df_for_writing["n_axis"].apply(np.ndarray.tobytes)

    with tempfile.TemporaryDirectory() as root_dir:
        logging.info("creating a fake directory")
        raw_dir = Path(root_dir).joinpath("raw")
        raw_dir.mkdir()

        raw_data_path = raw_dir.joinpath(filename + ".feather")

        df_for_writing.to_feather(raw_data_path)

        yield root_dir, filename
    logging.info("deleting test folder")


@pytest.fixture
def local_ray(scope="session"):
    ray.init(local_mode=True)
    yield
    ray.shutdown()


@pytest.fixture
def dataset(local_ray, dataset_root_and_filename, data_df):
    root_directory, data_filename = dataset_root_and_filename

    props = [
        "coord",
        "n_axis",
        "attachment_node_index",
        "attachment_angle",
        "attachment_block_index",
    ]

    dataset = BrutalDock(
        root_directory, props=props, file_names=[data_filename], proc_func=env3d_proc
    )
    return dataset


@pytest.fixture
def dataloader(local_ray, dataset):
    return DataLoader(dataset, shuffle=False, batch_size=3)