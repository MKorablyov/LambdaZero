from typing import Dict
from abc import abstractmethod

from torch import nn

from LambdaZero.datasets.brutal_dock.loggers.step_counter import StepCounter


class ExperimentLogger:
    def __init__(self, run_parameters: Dict[str, str], tracking_uri: str, tags: Dict[str, str], **kwargs):
        """

        Class to manage the internals of an experiment logger, for easy experiment logging.
        Args:
            run_parameters (Dict): dictionary containing the name of the experiment, the run name, and other relevant
            tracking_uri (str): path where the logger will write.
            tags (Dict): other tags
            kwargs : dictionary of parameters for specific logger.
        """
        super().__init__()
        self.step_counter = StepCounter()

    @abstractmethod
    def watch_model(self, model: nn.Module):
        pass

    @abstractmethod
    def log_configuration(self, config: Dict):
        pass

    @abstractmethod
    def log_artifact(self, path: str):
        pass

    @abstractmethod
    def finalize(self):
        pass

    @abstractmethod
    def _log_metrics(self, key: str, value: float, step: int):
        pass

    def increment_step_and_log_metrics(self, key, value):
        step = self.step_counter.increment_and_return_count()
        self._log_metrics(key, value, step)

    def log_metrics_at_current_step(self, key, value):
        step = self.step_counter.count
        self._log_metrics(key, value, step)

