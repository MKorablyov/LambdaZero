from typing import Dict

import mlflow
import logging


class StepCounter:
    """
    This class keeps track of how many times a run_id has been called.
    """

    def __init__(self):
        self._count = 0

    @property
    def count(self):
        return self._count

    def increment_and_return_count(self):
        self._count += 1
        return self._count


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
        self.step_counter = StepCounter()

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

        tags_with_reserved_names = self._create_tags_using_reserved_names(self.tags)
        run = self.mlflow_client.create_run(experiment_id=self.experiment_id, tags=tags_with_reserved_names)
        self._run_id = run.info.run_id
        return self._run_id

    def increment_step_and_log_metrics(self, key, value):
        step = self.step_counter.increment_and_return_count()
        self.mlflow_client.log_metric(self.run_id, key, value, step=step)

    def log_metrics_at_current_step(self, key, value):
        step = self.step_counter.count
        self.mlflow_client.log_metric(self.run_id, key, value, step=step)

    def log_parameters(self, prefix: str, parameters: Dict[str, str]):
        for key, value in parameters.items():
            self.mlflow_client.log_param(self.run_id, f"{prefix}--{key}", value)

    def finalize(self):
        self.mlflow_client.set_terminated(self.run_id, status='FINISHED')

    @classmethod
    def _create_tags_using_reserved_names(cls, tags: Dict[str, str]):
        """
        Use MLFlow specific reserved names so tags are presented in the correct place in the gui.
        For reserved names, see https: // www.mlflow.org / docs / latest / tracking.html  # tracking
        """

        git_hash = tags.pop("git_hash", 'none')
        execution_file_name = tags.pop("execution_file_name", 'none')
        user = tags.pop("user", 'none')
        run_name = tags.pop("run_name", 'none')

        tags_with_correct_names = dict(tags)
        tags_with_correct_names["mlflow.source.git.commit"] = git_hash
        tags_with_correct_names["mlflow.source.name"] = execution_file_name
        tags_with_correct_names["mlflow.user"] = user
        tags_with_correct_names["mlflow.runName"] = run_name
        return tags_with_correct_names





