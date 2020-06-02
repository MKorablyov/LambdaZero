from typing import Dict
from abc import abstractmethod

from torch import nn

from LambdaZero.loggers.step_counter import StepCounter


class ExperimentLogger:
    def __init__(self, run_parameters: Dict[str, str], tracking_uri: str, execution_filename: str, **kwargs):
        """

        Class to manage the internals of an experiment logger, for easy experiment logging.
        Args:
            run_parameters (Dict): dictionary containing the name of the experiment and the run name
            tracking_uri (str): path where the logger will write.
            execution_filename (str): name of the experiment file executed to start the experiment
            kwargs : dictionary of parameters for specific logger.
        """
        super().__init__()
        self._experiment_name = run_parameters.get("experiment_name", 'none')
        self._run_name = run_parameters.get("run_name", 'none')
        self._execution_filename = execution_filename
        self._tracking_uri = tracking_uri

        self.step_counter = StepCounter()

    @property
    def experiment_name(self):
        return self._experiment_name

    @property
    def run_name(self):
        return self._run_name

    @property
    def tracking_uri(self):
        return self._tracking_uri

    @property
    def execution_filename(self):
        return self._execution_filename

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

