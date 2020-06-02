import argparse
import json
import sys
from pathlib import Path

from LambdaZero import ROOT_DIR

CONFIG_KEY = "config"
MODEL_PARAMETERS_KEY = "model"
TRAINING_PARAMETERS_KEY = "training"
RUN_PARAMETERS_KEY = "run_parameters"

NON_CONFIG_KEY = "non_config"
PATHS_KEY = "paths"
EXECUTION_FILENAME_KEY = "execution_filename"


def get_input_arguments():
    parser = argparse.ArgumentParser(
        description="Run experiment and write output to specified work directory."
    )

    parser.add_argument(
        "--working_directory", help="Directory where to copy data to work"
    )

    parser.add_argument(
        "--output_directory", help="Directory where to copy results",
    )

    parser.add_argument(
        "--data_directory", help="Directory where to fetch preprocessed data",
    )

    parser.add_argument(
        "--tracking_uri", help="Directory where the experiment logger will store metric logs",
    )

    parser.add_argument("--config", help="path to input configuration file, in json format")

    args = parser.parse_args(sys.argv[1:])

    return args


def read_configuration_file(json_config_path: str):

    with open(json_config_path, "r") as f:
        input_config = json.load(f)

    return input_config


def default_json_writer(object):
    return object.__str__()


def write_configuration_file(json_config_path: str, config_dict: dict):
    with open(json_config_path, "w") as f:
        json.dump(config_dict, f, indent=4, default=default_json_writer)


def get_non_configuration_parameters(
    executable_file_path: Path,
    args: argparse.Namespace,
):
    paths = {"tracking_uri": args.tracking_uri,
             "working_directory": args.working_directory,
             "output_directory": args.output_directory,
             "data_directory":  args.data_directory,
             }

    non_config_dict = {PATHS_KEY: paths,
                       EXECUTION_FILENAME_KEY: str(executable_file_path.relative_to(ROOT_DIR))}

    return non_config_dict


def get_input_and_run_configuration(executable_file_path: Path):
    args = get_input_arguments()

    config = read_configuration_file(args.config)
    non_config = get_non_configuration_parameters(executable_file_path, args)

    config_and_augmented = {CONFIG_KEY: config,
                            NON_CONFIG_KEY: non_config}

    return config_and_augmented
