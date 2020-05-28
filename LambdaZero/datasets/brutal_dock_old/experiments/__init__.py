from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
EXPERIMENT_DATA_DIR = EXPERIMENT_DIR.joinpath('data')
RAW_EXPERIMENT_DATA_DIR = EXPERIMENT_DATA_DIR.joinpath('raw')
RAW_EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
