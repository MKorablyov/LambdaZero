import argparse
import json
import sys


def get_input_arguments():
    parser = argparse.ArgumentParser(
        description="Run env3d experiment."
    )

    parser.add_argument(
        "--data_directory", help="Directory where to fetch preprocessed data",
    )

    parser.add_argument("--config", help="path to input configuration file, in json format")

    args = parser.parse_args(sys.argv[1:])

    return args


def read_configuration_file(json_config_path: str):

    with open(json_config_path, "r") as f:
        input_config = json.load(f)

    return input_config
