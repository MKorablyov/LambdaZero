"""
This script aims to train a model. It is WIP.
"""
import shutil
import tempfile
from pathlib import Path

import ray

from LambdaZero.examples.env3d.dataset.processing import env3d_proc
from LambdaZero.inputs import BrutalDock
from LambdaZero.utils import get_external_dirs


datasets_dir, _, summaries_dir = get_external_dirs()
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")

data_filename_without_suffix = f"env3d_dataset_3_parent_blocks_debug"
data_filename = f"{data_filename_without_suffix}.feather"

source_path_to_dataset = results_dir.joinpath(data_filename)

props = ["coord", "n_axis", "attachment_node_index", "attachment_angle", "attachment_block_index"]

if __name__ == "__main__":

    ray.init(local_mode=True)

    with tempfile.TemporaryDirectory() as root_directory:
        #  Let's just sanity check that the dataset can be loaded
        raw_data_directory = Path(root_directory).joinpath('raw')
        raw_data_directory.mkdir()
        dest_path_to_dataset = raw_data_directory.joinpath(data_filename)
        shutil.copyfile(source_path_to_dataset, dest_path_to_dataset)

        dataset = BrutalDock(
            root_directory, props=props, file_names=[data_filename_without_suffix], proc_func=env3d_proc
        )

        print(f"size of dataset: {len(dataset)}")