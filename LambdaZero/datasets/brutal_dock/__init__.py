import logging.config
from pathlib import Path
import yaml

BRUTAL_DOCK_BASE_DIR = Path(__file__).parent

# TODO: some of these global variables should be defined in an init file closer to the actual root dir.
# We limit ourselves to coding here for now.
ROOT_DIR = BRUTAL_DOCK_BASE_DIR.parent.parent.parent

LOG_DIR = ROOT_DIR.joinpath("logs/")
LOG_DIR.mkdir(parents=True, exist_ok=True)


BRUTAL_DOCK_DATA_DIR = ROOT_DIR.joinpath("Datasets/brutal_dock/")

RESULTS_DIR = ROOT_DIR.joinpath("results/")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def set_logging_directory(logging_directory: Path):
    logging_path = ROOT_DIR.joinpath("logging.yaml")
    with open(logging_path, "rt") as f:
        config = yaml.safe_load(f.read())

    for log_type in ["info", "error"]:
        config["handlers"][f"{log_type}_file_handler"]["filename"] = str(
            logging_directory.joinpath(f"{log_type}.log")
        )
    logging.config.dictConfig(config)


set_logging_directory(LOG_DIR)

