from typing import Dict
from abc import abstractmethod

from torch import nn


class AbstactLogger:
    def __init__(self, run_parameters: Dict[str, str], tracking_uri: str, **kwargs):
        """

        Class to manage the internals of an experiment logger, for easy experiment logging.
        Args:
            run_parameters (Dict): dictionary containing the name of the experiment, the run name and other tags.
            tracking_uri (str): path where the logger will write.
            kwargs : dictionary of parameters for specific logger.
        """
        super().__init__()

    @abstractmethod
    def watch_model(self, model: nn.Module):
        pass

    @abstractmethod
    def log_metrics(self, key: str, value: float):
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


