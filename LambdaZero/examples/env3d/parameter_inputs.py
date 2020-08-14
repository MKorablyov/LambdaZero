import argparse
import json
import sys


def get_input_arguments():
    parser = argparse.ArgumentParser(
        description="Run env3d experiment."
    )

    parser.add_argument(
        "--root_path", help="Directory where to fetch data. Pytorch-geometric "
                            "assumes that the raw data is in [root_path]/raw/ and the processed data"
                            "is in [root_path]/processed/",
    )

    parser.add_argument(
        "--data_file_name", help="name of file with raw data. It is assumed that this is a pandas"
                                 "feather file."
    )

    parser.add_argument("--config", help="path to input configuration file, in json format")

    args = parser.parse_args(sys.argv[1:])

    return args


def read_configuration_file(json_config_path: str):

    with open(json_config_path, "r") as f:
        input_config = json.load(f)

    # json cannot read bool, so convert from str
    input_config["model_config"]["decouple_layers"] = str2bool(input_config["model_config"].get("decouple_layers", False))

    return input_config


def str2bool(input):
    if type(input) == bool:
        return input
    elif type(input) == int:
        # false if 0, true otherwise
        return input != 0 
    elif type(input) == str:
        if input.lower() in ["true", "t"]:
            return True
        elif input.lower() in ["false", "f"]:
            return False
        else:
            raise ValueError(f"{input} is an invalid string for decouple_layers argument")
    else:
        raise ValueError(f"{input} is an invalid format for decouple_layers. Should be bool, int or str.")
