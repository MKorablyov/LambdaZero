"""
This script computes the accuracy (blocks) and RMSE (angles) one can expect if we
define a baseline model that predicts a block using the normalized block frequency as
probability and draws an angle the marginal angle distribution.
"""
from pathlib import Path

import pandas as pd
import numpy as np

from LambdaZero.utils import get_external_dirs

datasets_dir, _, summaries_dir = get_external_dirs()

number_of_parent_blocks = 5

dataset_base_path = Path(summaries_dir).joinpath("env3d/dataset/from_cluster/RUN5/")
dataset_path = dataset_base_path.joinpath("data/raw/combined_dataset.feather")
results_dir = dataset_base_path.joinpath("analysis/")
results_dir.mkdir(exist_ok=True)

number_of_iterations = 100
if __name__ == "__main__":

    np.random.seed(0)

    df = pd.read_feather(dataset_path)

    normalized_counts = df["attachment_block_class"].value_counts(normalize=True)
    classes = np.array(normalized_counts.index)
    class_probabilities = normalized_counts.values

    number_of_elements = len(df)
    actual_classes = df["attachment_block_class"].values

    list_accuracy = []
    for _ in range(number_of_iterations):
        predicted_classes = np.random.choice(
            classes, len(actual_classes), p=class_probabilities
        )
        accuracy = np.sum(predicted_classes == actual_classes) / number_of_elements
        list_accuracy.append(accuracy)

    mean_accuracy = np.mean(list_accuracy)
    std_accuracy = np.std(list_accuracy)

    theoretical_accuracy = np.sum(class_probabilities ** 2)
    print(
        f" Mean accuracy = {mean_accuracy:4.3f} (sum_i p_i^2 = {theoretical_accuracy:4.3f})"
    )
    print(f" std accuracy = {std_accuracy:4.3f}")

    mask = df["attachment_angle"] >= 0.0
    actual_angles = df[mask]["attachment_angle"].values
    number_of_angle_elements = len(actual_angles)
    bins = np.linspace(0, 2 * np.pi, 361)  # one degree bin.
    hist, bin_edges = np.histogram(actual_angles, bins=bins)
    angle_probabilities = hist / np.sum(hist)

    angle_classes = (bins[1:] + bins[:-1]) / 2.0

    actual_sincos = np.stack([np.sin(actual_angles), np.cos(actual_angles)], axis=1)
    list_rmse = []
    for _ in range(number_of_iterations):

        predicted_angles = np.random.choice(
            angle_classes, number_of_angle_elements, p=angle_probabilities
        )
        predicted_sincos = np.stack(
            [np.sin(predicted_angles), np.cos(predicted_angles)], axis=1
        )

        total_square_error = np.sum((actual_sincos - predicted_sincos) ** 2, axis=1)
        rmse = np.sqrt(np.mean(total_square_error))
        list_rmse.append(rmse)

    mean_rmse = np.mean(list_rmse)
    std_rmse = np.std(list_rmse)

    # I'm pretty sure we can derive these numbers theoretically by doing some probability integrals,
    # and that the result wouldn't depend on the details of the probability distribution.
    # I just want a number, though, so this is faster.
    print(f" Mean RMSE = {mean_rmse:4.3f}")
    print(f" std RMSE = {std_rmse:4.3f}")
