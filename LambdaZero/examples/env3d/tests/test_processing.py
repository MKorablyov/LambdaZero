import logging
import tempfile
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import ray
from rdkit.Chem.rdmolfiles import MolFromSmiles

from LambdaZero.examples.env3d.dataset.processing import env3d_proc
from LambdaZero.inputs import BrutalDock


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


def test_env3d_proc(dataset_root_and_filename, data_df):

    root_directory, data_filename = dataset_root_and_filename

    props = [
        "coord",
        "n_axis",
        "attachment_node_index",
        "attachment_angle",
        "attachment_block_index",
    ]

    ray.init(local_mode=True)

    dataset = BrutalDock(
        root_directory, props=props, file_names=[data_filename], proc_func=env3d_proc
    )

    for property in [
        "attachment_node_index",
        "attachment_angle",
        "attachment_block_index",
    ]:
        expected_values = data_df[property].values
        computed_values = dataset.data[property].numpy()
        np.testing.assert_allclose(expected_values, computed_values)

    for row_index, graph in enumerate(dataset):
        row = data_df.iloc[row_index]

        expected_coords = row["coord"]
        computed_coords = graph.pos.numpy()
        np.testing.assert_allclose(expected_coords, computed_coords)

        expected_n_axis = row["n_axis"]
        computed_n_axis = graph["n_axis"].numpy()[0]
        np.testing.assert_allclose(expected_n_axis, computed_n_axis)
