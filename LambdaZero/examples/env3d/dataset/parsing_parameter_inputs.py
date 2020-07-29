import json
from typing import Dict


def extract_parameters_from_configuration_file(json_config_path: str):

    with open(json_config_path, "r") as f:
        input_config = json.load(f)

    expected_keys = {"debug_run", "num_cpus", "number_of_parent_blocks", "num_conf",
                     "max_iters", "max_number_of_molecules"}

    keys = set(input_config.keys())

    assert keys == expected_keys, "The config file does not contain the expected keys. Review input."

    return input_config


def get_output_filename(random_seed: int, config: Dict) -> str:
    number_of_parent_blocks = config["number_of_parent_blocks"]
    num_conf = config["num_conf"]
    max_iters = config["max_iters"]

    output_filename = f"env3d_dataset_" \
                      f"{number_of_parent_blocks}_parent_blocks_" \
                      f"num_conf_{num_conf}_" \
                      f"max_iters_{max_iters}_" \
                      f"master_random_seed_{random_seed}"

    if config["debug_run"]:
        output_filename += '_debug'

    output_filename += ".feather"

    return output_filename
