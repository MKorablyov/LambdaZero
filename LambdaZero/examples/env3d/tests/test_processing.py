from copy import copy

import numpy as np
import pytest
import torch

from LambdaZero.examples.env3d.dataset.processing import env3d_proc, transform_concatenate_positions_to_node_features
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


def test_transform_concatenate_positions_to_node_features(graph, transformed_graph):

    computed_transformed_graph = transform_concatenate_positions_to_node_features(graph)

    assert set(transformed_graph.keys) == set(computed_transformed_graph.keys)

    for prop in transformed_graph.keys:
        expected_tensor = transformed_graph[prop]
        computed_tensor = computed_transformed_graph[prop]
        assert torch.all(torch.eq(expected_tensor, computed_tensor))


def test_dataset_creation_with_transform(local_ray, dataset_root_and_filename, data_df):

    root_directory, data_filename = dataset_root_and_filename

    props = [
        "coord",
        "n_axis",
        "attachment_node_index",
        "attachment_angle",
        "attachment_block_index",
    ]

    dataset = BrutalDock(
        root_directory,
        props=props,
        file_names=[data_filename],
        proc_func=env3d_proc,
        transform=transform_concatenate_positions_to_node_features,
    )

    for graph in dataset:
        assert torch.all(torch.eq(graph.x[:, -3:], graph.pos))
