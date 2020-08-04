import numpy as np
import ray

from LambdaZero.examples.env3d.dataset.processing import env3d_proc
from LambdaZero.inputs import BrutalDock


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
