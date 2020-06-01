from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent
ANALYSIS_RESULTS_DIR = ANALYSIS_DIR.joinpath("results")
ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

