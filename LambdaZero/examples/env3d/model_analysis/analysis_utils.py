import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
from torch.nn import Softmax
from tqdm import tqdm


def get_truncated_confusion_matrix_and_labels(
    y_true: np.array, y_predicted: np.array, top_k: int = 3
):
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

    labels = np.concatenate([top_labels, ["other", "other match"]])

    cm = confusion_matrix(normalized_y_true, normalized_y_predicted, labels=labels)

    return cm, labels


def get_actual_and_predictions(model, training_dataloader):
    softmax = Softmax(dim=1)
    list_df = []
    with torch.no_grad():
        for batch in tqdm(training_dataloader):
            blocks_logits, angle_uv = model(batch)

            predicted_blocks = softmax(blocks_logits).argmax(dim=1).cpu().numpy()

            xy = angle_uv.cpu().numpy()
            x = xy[:, 0]
            y = xy[:, 1]
            predicted_angles = get_predicted_angle(x, y)
            df = pd.DataFrame(
                data={
                    "actual block": batch.attachment_block_class.detach().cpu().numpy(),
                    "actual angle": batch.attachment_angle.detach().cpu().numpy(),
                    "predicted block": predicted_blocks,
                    "predicted angle": predicted_angles,
                }
            )

            list_df.append(df)

    df = pd.concat(list_df)

    return df


def get_predicted_angle(x: np.array, y: np.array):
    angles = np.arctan2(y, x)
    return np.mod(angles, 2 * np.pi)
