import numpy as np
from sklearn.metrics import confusion_matrix


def get_truncated_confusion_matrix_and_labels(y_true: np.array, y_predicted: np.array, top_k: int = 3):
    number_of_values = len(y_true)
    assert number_of_values == len(y_predicted), "the array size are not the same"

    unique_true, true_counts = np.unique(y_true, return_counts=True)
    sorting_indices = np.argsort(true_counts)[::-1]
    unique_true = unique_true[sorting_indices]

    top_labels = unique_true[:top_k]

    m_true_top_k = np.isin(y_true, top_labels)
    m_pred_top_k = np.isin(y_predicted, top_labels)
    m_equal = y_true == y_predicted

    tmp_true = np.where(m_true_top_k, y_true, "other")
    tmp_pred = np.where(m_pred_top_k, y_predicted, "other")
    other_match = np.where(m_equal, "other match", "placeholder")

    mask_true = np.logical_or(m_true_top_k, np.logical_not(m_equal))
    normalized_y_true = np.where(mask_true, tmp_true, other_match)

    mask_pred = np.logical_or(m_pred_top_k, np.logical_not(m_equal))
    normalized_y_predicted = np.where(mask_pred, tmp_pred, other_match)

    labels = np.concatenate([top_labels, ['other', 'other match']])

    cm = confusion_matrix(normalized_y_true,
                          normalized_y_predicted,
                          labels=labels)

    return cm, labels