import logging.config
from pathlib import Path

import yaml

# Define the path to the project's root directory.
ROOT_DIR = Path(__file__).parent.parent

LOG_DIR = ROOT_DIR.joinpath("logs/")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_logging_directory(logging_directory: Path):
    logging_path = ROOT_DIR.joinpath("logging.yaml")
    with open(logging_path, "rt") as f:
        config = yaml.safe_load(f.read())

    for log_type in ["info", "error", "debug"]:
        config["handlers"][f"{log_type}_file_handler"]["filename"] = str(
            logging_directory.joinpath(f"{log_type}.log")
        )
    logging.config.dictConfig(config)


set_logging_directory(LOG_DIR)
