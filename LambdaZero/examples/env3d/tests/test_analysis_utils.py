import numpy as np

from LambdaZero.examples.env3d.model_analysis.analysis_utils import get_truncated_confusion_matrix_and_labels


def test_get_truncated_confusion_matrix_and_labels():

    top_k = 3
    y_true = np.array([0, 0, 0, 1, 1, 5, 5, 5, 5, 5, 5, 5, 27, 11])
    y_predicted = np.array([1, 0, 0, 1, 1, 5, 5, 3, 9, 0, 1, 5, 27, 12])

    expected_labels = ['5', '0', '1', 'other', 'other match']

    expected_cm = np.array([[3, 1, 1, 2, 0],
                            [0, 2, 1, 0, 0],
                            [0, 0, 2, 0, 0],
                            [0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 1]])

    computed_cm, computed_labels = get_truncated_confusion_matrix_and_labels(y_true,
                                                                             y_predicted,
                                                                             top_k)

    np.testing.assert_array_equal(computed_cm, expected_cm)
    np.testing.assert_array_equal(expected_labels, computed_labels)
