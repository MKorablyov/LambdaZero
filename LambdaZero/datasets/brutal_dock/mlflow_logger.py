from typing import Dict

import mlflow
import logging


class MLFlowLogger:
    def __init__(self, experiment_name: str, tracking_uri: str, tags: Dict[str, str]):
        """

        Class to manage the internals of mlflow, for easy experiment logging.
        Args:
            experiment_name (str): The name of the experiment
            tracking_uri (str): path where the logger will write.
            tags (dict): job identifiers
        """
        super().__init__()
        self.mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
        self.experiment_name = experiment_name
        self.tags = tags
        self._run_id = None

    @property
    def experiment_id(self):
        experiment = self.mlflow_client.get_experiment_by_name(self.experiment_name)

        if experiment:
             return experiment.experiment_id
        else:
            logging.warning(f"Experiment with name {self.experiment_name} not found. Creating it.")
            return self.mlflow_client.create_experiment(name=self.experiment_name)

    @property
    def run_id(self):
        if self._run_id is not None:
            return self._run_id
        run = self.mlflow_client.create_run(experiment_id=self.experiment_id, tags=self.tags)
        self._run_id = run.info.run_id
        return self._run_id

    def log_metrics(self, key, value, step):
        self.mlflow_client.log_metric(self.run_id, key, value, step)

    def finalize(self):
        self.mlflow_client.set_terminated(self.run_id, status='FINISHED')

