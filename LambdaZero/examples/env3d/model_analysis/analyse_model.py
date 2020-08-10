import logging
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from torch_geometric.data import DataLoader

from LambdaZero.examples.dataset_splitting import RandomDatasetSplitter
from LambdaZero.examples.env3d.dataset import ENV3D_DATA_PROPERTIES
from LambdaZero.examples.env3d.dataset.processing import (
    env3d_proc,
    transform_concatenate_positions_to_node_features,
)
from LambdaZero.examples.env3d.model_analysis.analysis_utils import (
    get_truncated_confusion_matrix_and_labels,
    get_actual_and_predictions,
)
from LambdaZero.examples.env3d.models.joint_prediction_model import BlockAngleModel
from LambdaZero.inputs import BrutalDock
from LambdaZero.utils import get_external_dirs

sns.set(font_scale=1.5)

model_dir = Path("/Users/bruno/Desktop/model/")
model_path = model_dir.joinpath("model.pth")

model_config = {
    "num_angle_prediction_hidden_features": 128,
    "num_block_prediction_hidden_features": 256,
    "num_edge_network_hidden_features": 32,
    "num_hidden_features": 64,
    "number_of_block_classes": 105,
    "number_of_layers": 4,
    "set2set_steps": 3,
}

dataset_config = {
    "root": "/Users/bruno/LambdaZero/summaries/env3d/dataset/from_cluster/RUN4/data",
    "props": ENV3D_DATA_PROPERTIES,
    "proc_func": env3d_proc,
    "transform": transform_concatenate_positions_to_node_features,
    "file_names": ["combined_dataset"],
}

_, _, summaries_dir = get_external_dirs()

dataset_pickle_path = Path(summaries_dir).joinpath("debug/forecast.pkl")
dataset_pickle_path.parent.mkdir(exist_ok=True)

if __name__ == "__main__":
    if not dataset_pickle_path.is_file():
        # generate the forecast if it doesn't exist already
        # This is DANGEROUS. It is VERY EASY to forget to delete the pickle
        # to regenerate. Handle with care!
        model = BlockAngleModel(**model_config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        dataset = BrutalDock(**dataset_config)
        dataset_splitter = RandomDatasetSplitter(
            train_fraction=0.8, validation_fraction=0.1, random_seed=0
        )

        training_dataset, _, _ = dataset_splitter.get_split_datasets(dataset)

        batchsize = 256
        training_dataloader = DataLoader(
            training_dataset, shuffle=True, batch_size=batchsize
        )

        logging.info("Creating the forecast pickle.")
        df = get_actual_and_predictions(model, training_dataloader)
        df.to_pickle(dataset_pickle_path)

    logging.info("Reading pickle.")
    df = pd.read_pickle(dataset_pickle_path)

    # Plot confusion matrix
    cm, display_labels = get_truncated_confusion_matrix_and_labels(
        y_true=df["actual block"].values,
        y_predicted=df["predicted block"].values,
        top_k=8,
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Block index confusion matrix")
    ax = fig.add_subplot(111)
    ax.grid(False)
    disp.plot(
        include_values=True,
        cmap="viridis",
        ax=ax,
        xticks_rotation="horizontal",
        values_format=None,
    )

    # Plot angle error
    angle_df = copy(df[df["actual angle"] >= 0.0])

    for kind in ["actual", "predicted"]:
        angle_df[f"{kind} cos"] = angle_df[f"{kind} angle"].apply(np.cos)
        angle_df[f"{kind} sin"] = angle_df[f"{kind} angle"].apply(np.sin)

    actual_vectors = angle_df[["actual cos", "actual sin"]].values
    predicted_vectors = angle_df[["predicted cos", "predicted sin"]].values

    z = np.array([0.0, 0.0, 1.0])

    # treat the actual vectors as the x axis in their reference frame. The y axis is then given by
    y_direction = np.cross(z, actual_vectors)[:, :2]

    x = np.einsum("ij, ij->i", actual_vectors, predicted_vectors)
    y = np.einsum("ij, ij->i", y_direction, predicted_vectors)
    angle_error = np.arctan2(y, x)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Angle error")
    ax = fig.add_subplot(111)
    ax.hist(angle_error / np.pi * 180.0, bins=60)
    ax.set_xlabel("Angle error, in degree")
    ax.set_ylabel("count")
    ax.set_xlim([-180.0, 180.0])

    plt.show()
