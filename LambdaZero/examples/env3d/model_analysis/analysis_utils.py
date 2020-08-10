import numpy as np
from sklearn.metrics import confusion_matrix


def get_truncated_confusion_matrix_and_labels(y_true: np.array, y_predicted: np.array, top_k: int = 3):
    unique_true, true_counts = np.unique(y_true, return_counts=True)
    sorting_indices = np.argsort(true_counts)[::-1]
    unique_true = unique_true[sorting_indices]

    top_labels = unique_true[:top_k]

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

    labels = np.concatenate([top_labels, ['other', 'other match']])

    cm = confusion_matrix(normalized_y_true,
                          normalized_y_predicted,
                          labels=labels)

    return cm, labels