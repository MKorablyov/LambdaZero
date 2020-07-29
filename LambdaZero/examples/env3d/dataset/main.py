"""
The goal of this script is to generate the dataset of molecules embedded in 3D space.
"""
import argparse
import json
import os
import sys

from pathlib import Path

import numpy as np
import ray
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import LambdaZero.utils
from LambdaZero.examples.env3d.dataset.data_row_generator import DataRowGenerator
from LambdaZero.examples.env3d.dataset.io_utilities import (
    create_or_append_feather_file,
    get_debug_blocks,
)

# This script assumes LambdaZero is properly installed, and that the
# external dependencies are available.
from LambdaZero.examples.env3d.dataset.parsing_parameter_inputs import (
    extract_parameters_from_configuration_file,
    get_output_filename,
)

datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")
results_dir.mkdir(exist_ok=True, parents=True)

if __name__ == "__main__":
    # Parse script input
    parser = argparse.ArgumentParser(
        description="Generate env3d dataset and write output to specified output path."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="path to input configuration file, in json format",
    )

    args = parser.parse_args(sys.argv[1:])
    config = extract_parameters_from_configuration_file(args.config)

    if config["debug_run"]:  # Small parameters for quick execution
        debug_blocks = get_debug_blocks(blocks_file, number_of_blocks=10)
        # overload the block file with a small subset, for quick execution
        blocks_file = results_dir.joinpath("blocks_debug.json")
        with open(blocks_file, "w") as f:
            json.dump(debug_blocks, f)

    output_file_name = get_output_filename(config)
    output_path = results_dir.joinpath(output_file_name)

    num_cpus = config["num_cpus"]
    max_number_of_molecules = config["max_number_of_molecules"]

    ray.init(num_cpus=num_cpus)

    np.random.seed(config["master_random_seed"])

    generators = []
    for _ in range(num_cpus):
        # Generate a ray Actor for each available cpu. It gets its own random seed to make
        # sure actors are not clones of each other.
        random_seed = np.random.randint(1e9)
        g = DataRowGenerator.remote(
            blocks_file,
            config["number_of_parent_blocks"],
            config["num_conf"],
            config["max_iters"],
            random_seed,
        )
        generators.append(g)

    # round robin the tasks to the different ray actors
    row_ids = [
        generators[i % num_cpus].generate_row.remote()
        for i in range(max_number_of_molecules)
    ]

    done_count = 0
    while row_ids:
        done_ids, row_ids = ray.wait(row_ids)
        done_count += len(done_ids)
        print(f"Done {done_count} out of {max_number_of_molecules}")

        for done_id in done_ids:
            try:
                byte_row = ray.get(done_id)
                create_or_append_feather_file(output_path, byte_row)
            except ValueError as e:
                print("Something went wrong with molecule generation. Exception:")
                print(e)
                print("Moving on.")

    ray.shutdown()
