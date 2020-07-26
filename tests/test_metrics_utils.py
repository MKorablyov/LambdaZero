import numpy as np

from LambdaZero.datasets.temp_brunos_work.metrics_utils import get_prediction_statistics


def test_get_prediction_statistics():
    size = 50

    np.random.seed(23423)
    list_actuals = np.random.random(size)
    list_predicted = np.random.random(size)

    errors = np.abs(list_actuals-list_predicted)

    expected_mean = np.sum(errors)/size

    var = np.sum(errors**2)/size - expected_mean**2
    expected_std = np.sqrt(var)

    computed_mean, computed_std = get_prediction_statistics(list_actuals, list_predicted)

    np.testing.assert_almost_equal(expected_mean, computed_mean)
    np.testing.assert_almost_equal(computed_std, expected_std)
