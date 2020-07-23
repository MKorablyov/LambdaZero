"""
The goal of this script is to generate the dataset of molecules embedded in 3D space.
"""
import os
from pathlib import Path

import numpy as np

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.dataset.data_generation import get_data_row
from LambdaZero.examples.env3d.dataset.io_utilities import (
    process_row_for_writing_to_feather,
    create_or_append_feather_file,
)

datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")

results_dir.mkdir(exist_ok=True)

number_of_parent_blocks = 5

output_path = results_dir.joinpath(
    f"env3d_dataset_{number_of_parent_blocks}_parent_blocks.feather"
)

num_conf = 25
max_iters = 200
random_seed = 12312

max_number_of_molecules = 1000

if __name__ == "__main__":
    np.random.seed(random_seed)

    reference_molMDP = MolMDP(blocks_file=blocks_file)

    for index in range(max_number_of_molecules):
        print(f" Computing molecule {index} of {max_number_of_molecules}")
        reference_molMDP.reset()
        reference_molMDP.random_walk(number_of_parent_blocks)
        number_of_stems = len(reference_molMDP.molecule.stems)

        if number_of_stems < 1:
            print("no stems! Moving on to next molecule")
            continue

        attachment_stem_idx = np.random.choice(number_of_stems)

        try:
            row = get_data_row(
                reference_molMDP, attachment_stem_idx, num_conf, max_iters, random_seed
            )
        except:
            print("Something went wrong while relaxing molecule. Moving on to next molecule")
            continue

        byte_row = process_row_for_writing_to_feather(row)
        create_or_append_feather_file(output_path, byte_row)
