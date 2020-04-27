from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parent
EXPERIMENT_DATA_DIR = EXPERIMENT_DIR.joinpath('data')
EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
