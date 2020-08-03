"""
The goal of this script is to generate the dataset of molecules embedded in 3D space.

This script assumes LambdaZero is properly installed, and that the external dependencies are available,
namely that the script "install-prog-data.sh" has been executed so that the method get_external_dirs()
can fetch the correct set of external directories.
"""
import logging
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import ray

from LambdaZero.examples.env3d.dataset.data_row_generator import DataRowGenerator
from LambdaZero.examples.env3d.dataset.io_utilities import (
    create_or_append_feather_file,
    get_debug_blocks,
)
from LambdaZero.examples.env3d.dataset.parsing_parameter_inputs import (
    extract_parameters_from_configuration_file,
    get_output_filename,
)
from LambdaZero.utils import get_external_dirs

datasets_dir, _, summaries_dir = get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")
results_dir.mkdir(exist_ok=True, parents=True)
logging_dir = results_dir.joinpath("logs")
logging_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

megabyte = 1024 * 1024

if __name__ == "__main__":
    # Parse script input
    parser = argparse.ArgumentParser(
        description="Generate env3d dataset and write output to specified output path."
    )

    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="master random seed, used to generate Actor random seeds, int",
    )

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="path to input configuration file, in json format",
    )

    args = parser.parse_args(sys.argv[1:])
    master_random_seed = args.seed
    np.random.seed(master_random_seed)

    log_file_name = str(
        logging_dir.joinpath(f"info_MASTER_seed_{master_random_seed}.log")
    )

    logging.basicConfig(filename=log_file_name,
                        format="%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - %(message)s",
                        filemode="w",
                        level=logging.INFO)

    config = extract_parameters_from_configuration_file(args.config)

    if config["debug_run"]:  # Small parameters for quick execution
        debug_blocks = get_debug_blocks(blocks_file, number_of_blocks=5)
        # overload the block file with a small subset, for quick execution
        blocks_file = results_dir.joinpath("blocks_debug.json")
        with open(blocks_file, "w") as f:
            json.dump(debug_blocks, f)

    num_cpus = config["num_cpus"]
    max_number_of_molecules = config["max_number_of_molecules"]

    logger.info("Initializing Ray")
    ray.init(
        num_cpus=num_cpus,
        memory=500 * megabyte,
        object_store_memory=300 * megabyte,
        driver_object_store_memory=200 * megabyte,
    )

    generators = []
    output_file_paths = []
    for _ in range(num_cpus):
        # Generate a ray Actor for each available cpu. It gets its own random seed to make
        # sure actors are not clones of each other.
        random_seed = np.random.randint(1e6)

        logger.info(f"Initializing Actor with seed {random_seed}")
        g = DataRowGenerator.remote(
            blocks_file,
            config["number_of_parent_blocks"],
            config["num_conf"],
            config["max_iters"],
            random_seed,
            logging_dir,
        )
        generators.append(g)

        # We'll create one output file per ray Actor, for reproducibility.
        output_path = results_dir.joinpath(get_output_filename(random_seed, config))
        output_file_paths.append(output_path)

    # round robin the tasks to the different ray actors
    row_ids = []
    output_file_path_dict = dict()
    for i in range(max_number_of_molecules):
        actor_index = i % num_cpus
        row_id = generators[actor_index].generate_row.remote()
        row_ids.append(row_id)
        output_file_path_dict[row_id] = output_file_paths[actor_index]

    done_count = 0
    while row_ids:
        #  This "while" mechanism is ray's way of getting results as soon as they are ready.
        done_ids, row_ids = ray.wait(row_ids)
        done_count += len(done_ids)
        logger.info(f"Done {done_count} out of {max_number_of_molecules}")

        for done_id in done_ids:
            try:
                byte_row = ray.get(done_id)
                output_path = output_file_path_dict[done_id]
                create_or_append_feather_file(output_path, byte_row)
            except (ValueError, AssertionError) as e:
                logger.warning(
                    "Something went wrong with molecule generation. Exception:"
                )
                logger.warning(e)
                logger.warning("Moving on.")

    ray.shutdown()
