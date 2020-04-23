from pathlib import Path

BRUTAL_DOCK_BASE_DIR = Path(__file__).parent

# TODO: this should be defined in an init file closer to the actual root dir.
# We limit ourselves to coding here for now.
ROOT_DIR = BRUTAL_DOCK_BASE_DIR.parent.parent.parent

BRUTAL_DOCK_DATA_DIR = ROOT_DIR.joinpath("Datasets/brutal_dock/")
