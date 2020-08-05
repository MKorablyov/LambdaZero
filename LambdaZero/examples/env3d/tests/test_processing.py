from copy import copy

import numpy as np
import pytest
import torch
from rdkit.Chem.rdmolfiles import MolFromSmiles

from LambdaZero.examples.env3d.dataset import ENV3D_DATA_PROPERTIES
from LambdaZero.examples.env3d.dataset.processing import (
    env3d_proc,
    transform_concatenate_positions_to_node_features,
)
from LambdaZero.inputs import BrutalDock


@pytest.fixture
def graph(dataset):
    return dataset[0]


@pytest.fixture
def transformed_graph(graph):

    new_graph = copy(graph)

    new_x = []
    for node_index in range(graph.num_nodes):
        new_x.append(torch.cat([new_graph.x[node_index], new_graph.pos[node_index]]))
    new_graph.x = torch.stack(new_x)

    return new_graph


def test_env3d_proc(local_ray, dataset_root_and_filename, data_df):

    root_directory, data_filename = dataset_root_and_filename

    dataset = BrutalDock(
        root_directory,
        props=ENV3D_DATA_PROPERTIES,
        file_names=[data_filename],
        proc_func=env3d_proc,
    )

    for property in [
        "attachment_node_idx",
        "attachment_angle",
        "attachment_block_class",
    ]:
        expected_values = data_df[property].values
        computed_values = dataset.data[property].numpy()
        np.testing.assert_allclose(expected_values, computed_values)

    for row_index, graph in enumerate(dataset):
        row = data_df.iloc[row_index]

        computed_attachment_node_index = graph.attachment_node_idx.numpy()[0]
        expected_attachment_node_index = row["attachment_node_idx"]

        assert computed_attachment_node_index == expected_attachment_node_index

        expected_coords = row["coord"]
        computed_coords = graph.pos.numpy()
        np.testing.assert_allclose(expected_coords, computed_coords)

        expected_n_axis = row["n_axis"]
        computed_n_axis = graph["n_axis"].numpy()[0]
        np.testing.assert_allclose(expected_n_axis, computed_n_axis)


def test_transform_concatenate_positions_to_node_features(graph, transformed_graph):

    computed_transformed_graph = transform_concatenate_positions_to_node_features(graph)

    assert set(transformed_graph.keys) == set(computed_transformed_graph.keys)

    for prop in transformed_graph.keys:
        expected_tensor = transformed_graph[prop]
        computed_tensor = computed_transformed_graph[prop]
        assert torch.all(torch.eq(expected_tensor, computed_tensor))


def test_dataset_creation_with_transform(local_ray, dataset_root_and_filename, data_df):

    root_directory, data_filename = dataset_root_and_filename

    dataset = BrutalDock(
        root_directory,
        props=ENV3D_DATA_PROPERTIES,
        file_names=[data_filename],
        proc_func=env3d_proc,
        transform=transform_concatenate_positions_to_node_features,
    )

    for graph in dataset:
        assert torch.all(torch.eq(graph.x[:, -3:], graph.pos))


def test_dataloader_sanity(data_df, dataloader, dataset):

    list_atom_counts = (
        data_df["smi"].apply(lambda s: MolFromSmiles(s).GetNumAtoms()).values
    )
    attachment_angles = []
    attachment_node_indices = []
    attachment_block_indices = []

    for batch in dataloader:
        attachment_angles.append(batch.attachment_angle)
        attachment_node_indices.append(batch.attachment_node_idx)
        attachment_block_indices.append(batch.attachment_block_class)

    computed_attachment_angles = torch.cat(attachment_angles)
    expected_attachment_angles = torch.tensor(
        data_df["attachment_angle"].values, dtype=torch.float
    )
    assert torch.allclose(expected_attachment_angles, computed_attachment_angles)

    computed_attachment_node_indices = torch.cat(attachment_node_indices)
    expected_attachment_node_indices = torch.tensor(
        data_df["attachment_node_idx"].values
    )
    assert torch.allclose(
        computed_attachment_node_indices, expected_attachment_node_indices
    )

    computed_attachment_block_indices = torch.cat(attachment_block_indices)
    expected_attachment_block_indices = torch.tensor(
        data_df["attachment_block_class"].values
    )
    assert torch.allclose(
        computed_attachment_block_indices, expected_attachment_block_indices
    )
