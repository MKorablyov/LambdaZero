from typing import Dict

from LambdaZero.core.alpha_zero_trainer import nn
from LambdaZero.datasets.brutal_dock.loggers.experiment_logger import ExperimentLogger


class NullLogger(ExperimentLogger):
    def __init__(self, run_parameters: Dict[str, str], tracking_uri: str, execution_filename: str, **kwargs):
        """
        Empty logger that does nothing. This is useful for debugging.
        """
        super().__init__(run_parameters, tracking_uri, execution_filename)

    def watch_model(self, model: nn.Module):
        pass

    def log_configuration(self, config: Dict):
        pass

    def log_artifact(self, path: str):
        pass

    def finalize(self):
        pass

    def _log_metrics(self, key: str, value: float, step: int):
        pass
