import argparse
import json
import sys
from pathlib import Path
import getpass

import git

from LambdaZero.datasets.brutal_dock import ROOT_DIR

MODEL_PARAMETERS_KEY = "model"
TRAINING_PARAMETERS_KEY = "training"
RUN_PARAMETERS_KEY = "run_parameters"


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


def get_git_hash():
    repo = git.Repo(ROOT_DIR)
    git_hash = repo.head.object.hexsha
    return git_hash


def get_user():
    return getpass.getuser()


def default_json_writer(object):
    return object.__str__()


def write_configuration_file(json_config_path: str, config_dict: dict):
    with open(json_config_path, "w") as f:
        json.dump(config_dict, f, indent=4, default=default_json_writer)


def augment_configuration_with_run_parameters(
    input_config: dict,
    executable_file_path: Path,
    args: argparse.ArgumentParser,
):
    input_and_run_config = dict(input_config)

    run_parameters_dict = input_and_run_config[RUN_PARAMETERS_KEY]
    run_parameters_dict["tracking_uri"] = args.tracking_uri

    run_parameters_dict["git_hash"] = get_git_hash()
    run_parameters_dict["user"] = get_user()

    run_parameters_dict["working_directory"] = args.working_directory
    run_parameters_dict["output_directory"] = args.output_directory
    run_parameters_dict["data_directory"] = args.data_directory

    run_parameters_dict["execution_file_name"] = str(executable_file_path.relative_to(ROOT_DIR))

    input_and_run_config[RUN_PARAMETERS_KEY] = run_parameters_dict

    return input_and_run_config


def get_input_and_run_configuration(executable_file_path: Path):
    args = get_input_arguments()

    input_config = read_configuration_file(args.config)

    input_and_run_config = augment_configuration_with_run_parameters(
        input_config,
        executable_file_path,
        args)

    return input_and_run_config
