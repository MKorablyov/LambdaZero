import mlflow
import numpy as np

import pytest

from LambdaZero.datasets.temp_brunos_work.loggers.mlflow_logger import MLFlowLogger
from LambdaZero.datasets.temp_brunos_work.loggers.step_counter import StepCounter


@pytest.fixture
def run_parameters(experiment_name):
    return {'experiment_name': experiment_name, 'run_name': 'TEST'}


@pytest.fixture
def execution_filename():
    return "some_test_file_name.py"


@pytest.fixture
def metrics():
    number_of_metrics = 3
    np.random.seed(123)
    x = np.random.rand(number_of_metrics)
    return x


@pytest.fixture
def key():
    return 'test_metric'


@pytest.fixture
def mlflow_logger_with_logging(run_parameters, tracking_uri, execution_filename, key, metrics):

    mlflow_logger = MLFlowLogger(run_parameters, tracking_uri, execution_filename)

    for step, value in enumerate(metrics):
        mlflow_logger.increment_step_and_log_metrics(key, value)

    return mlflow_logger


def test_mlflow_logger_name(mlflow_logger_with_logging, experiment_name, tracking_uri):
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = mlflow_client.get_experiment(mlflow_logger_with_logging.experiment_id)
    assert experiment.name == experiment_name


def test_mlflow_logger_tags(mlflow_logger_with_logging, tracking_uri):
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    run = mlflow_client.get_run(mlflow_logger_with_logging.run_id)
    tags_with_reserved_names = mlflow_logger_with_logging._create_tags_using_reserved_names()
    assert run.data.tags == tags_with_reserved_names


def test_mlflow_logger_metrics(mlflow_logger_with_logging, key, metrics, tracking_uri):
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    metric_history = mlflow_client.get_metric_history(mlflow_logger_with_logging.run_id, key)
    computed_metrics = np.array([m.value for m in metric_history])
    np.testing.assert_array_equal(computed_metrics, metrics)


@pytest.mark.parametrize("finalize, status", [(False, "RUNNING"), (True, "FINISHED")])
def test_mlflow_logger_status(mlflow_logger_with_logging, tracking_uri, finalize, status):

    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    if finalize:
        mlflow_logger_with_logging.finalize()

    run = mlflow_client.get_run(mlflow_logger_with_logging.run_id)
    assert run.info.status == status


def test_step_counter():
    step_counter = StepCounter()
    for i in range(1, 11):
        assert i == step_counter.increment_and_return_count()

    assert step_counter.count == 10

