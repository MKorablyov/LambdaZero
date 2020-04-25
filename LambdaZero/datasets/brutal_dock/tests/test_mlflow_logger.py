import logging
import tempfile

import mlflow
import numpy as np

import pytest

from LambdaZero.datasets.brutal_dock.mlflow_logger import MLFlowLogger


@pytest.fixture
def experiment_name():
    return 'some-fake-experiment-name'


@pytest.yield_fixture
def tracking_uri():
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        logging.info("creating a fake directory")
        yield tmp_dir_str
    logging.info("deleting test folder")


@pytest.fixture
def tags():
    return {"id": "abcdef", "other-test-thing": '123'}


@pytest.fixture
def metrics():
    number_of_metrics = 3
    np.random.seed(123)
    x = np.random.rand(number_of_metrics)
    return x


def test_mlflow_logger(tmpdir, experiment_name, tracking_uri, tags, metrics):

    mlflow_logger = MLFlowLogger(experiment_name, tracking_uri, tags)

    key = 'test_metric'
    for step, value in enumerate(metrics):
        mlflow_logger.log_metrics(key, value,  step)

    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment = mlflow_client.get_experiment(mlflow_logger.experiment_id)
    assert experiment.name == experiment_name

    run = mlflow_client.get_run(mlflow_logger.run_id)
    assert run.data.tags == tags

    metric_history = mlflow_client.get_metric_history(mlflow_logger.run_id, key)
    computed_metrics = np.array([m.value for m in metric_history])

    np.testing.assert_array_equal(computed_metrics, metrics)





