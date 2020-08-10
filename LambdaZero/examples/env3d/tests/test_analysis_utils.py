import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from LambdaZero.examples.env3d.model_analysis.analysis_utils import get_truncated_confusion_matrix_and_labels


@pytest.fixture
def top_k():
    return 3


@pytest.fixture
def y_true_and_y_predicted():
    np.random.seed(2342)
    number_of_value = 100
    probabilities = [0.4, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05]
    y_true = np.random.choice(range(7), number_of_value, p=probabilities)
    y_predicted = np.random.choice(range(7), number_of_value, p=probabilities)
    return y_true, y_predicted


@pytest.fixture
def expected_confusion_matrix_and_labels(y_true_and_y_predicted, top_k):
    y_true, y_predicted = y_true_and_y_predicted

    unique_true, true_counts = np.unique(y_true, return_counts=True)
    sorting_indices = np.argsort(true_counts)[::-1]
    unique_true = unique_true[sorting_indices]
    top_labels = unique_true[:top_k]
    expected_labels = np.concatenate([top_labels, ['other', 'other match']])

    normalized_y_true = []
    normalized_y_predicted = []
    for a, p in zip(y_true, y_predicted):
        if a in top_labels:
            a_label = a
            if p in top_labels:
                p_label = p
            else:
                p_label = 'other'
        else:
            if p in top_labels:
                a_label = 'other'
                p_label = p
            else:
                if a == p:
                    a_label = 'other match'
                    p_label = 'other match'
                else:
                    p_label = 'other'
                    a_label = 'other'

        normalized_y_true.append(a_label)
        normalized_y_predicted.append(p_label)

    cm = confusion_matrix(normalized_y_true,
                          normalized_y_predicted,
                          labels=expected_labels)

    return cm, expected_labels


def test_get_truncated_confusion_matrix_and_labels(y_true_and_y_predicted, expected_confusion_matrix_and_labels, top_k):

    expected_cm, expected_labels = expected_confusion_matrix_and_labels

    y_true, y_predicted = y_true_and_y_predicted
    computed_cm, computed_labels = get_truncated_confusion_matrix_and_labels(y_true,
                                                                             y_predicted,
                                                                             top_k)

    np.testing.assert_array_equal(computed_cm, expected_cm)
    np.testing.assert_array_equal(expected_labels, computed_labels)
