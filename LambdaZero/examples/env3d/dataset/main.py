"""
The goal of this script is to generate the dataset of molecules embedded in 3D space.
"""
import json
import os
from pathlib import Path

import numpy as np
import ray

import LambdaZero.utils
from LambdaZero.examples.env3d.dataset.data_row_generator import DataRowGenerator
from LambdaZero.examples.env3d.dataset.io_utilities import (
    create_or_append_feather_file, get_debug_blocks,
)

datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")
results_dir.mkdir(exist_ok=True, parents=True)

debug_flag = False

master_random_seed = 12312
num_cpus = 12

if debug_flag:  # Small parameters for quick execution
    number_of_parent_blocks = 3
    number_of_blocks = 10
    num_conf = 50
    max_iters = 200
    max_number_of_molecules = 24
    output_filename = f"env3d_dataset_{number_of_parent_blocks}_parent_blocks_debug.feather"

    debug_blocks = get_debug_blocks(blocks_file, number_of_blocks)
    blocks_file = results_dir.joinpath("blocks_debug.json")
    with open(blocks_file, 'w') as f:
        json.dump(debug_blocks, f)

else:
    number_of_parent_blocks = 5
    num_conf = 25
    max_iters = 200
    max_number_of_molecules = 1000
    output_filename = f"env3d_dataset_{number_of_parent_blocks}_parent_blocks.feather"

output_path = results_dir.joinpath(output_filename)

if __name__ == "__main__":

    ray.init(num_cpus=num_cpus)

    np.random.seed(master_random_seed)

    generators = []

    for _ in range(num_cpus):
        random_seed = np.random.randint(1e9)
        g = DataRowGenerator.remote(blocks_file, number_of_parent_blocks, num_conf, max_iters, random_seed)
        generators.append(g)

    row_ids = [generators[i % num_cpus].generate_row.remote() for i in range(max_number_of_molecules)]

    done_count = 0
    while row_ids:
        done_ids, row_ids = ray.wait(row_ids)
        done_count += len(done_ids)
        print(f"Done {done_count} out of {max_number_of_molecules}")

        for done_id in done_ids:
            try:
                byte_row = ray.get(done_id)
                create_or_append_feather_file(output_path, byte_row)
            except:
                print('Something went wrong with molecule generation. Moving on.')
