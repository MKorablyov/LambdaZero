"""
TODO : description
"""
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from LambdaZero.datasets.brutal_dock import ROOT_DIR, RESULTS_DIR, BRUTAL_DOCK_DATA_DIR
from LambdaZero.datasets.brutal_dock.dataset_utils import get_scores_statistics
from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.experiments import EXPERIMENT_DATA_DIR, RAW_EXPERIMENT_DATA_DIR
from LambdaZero.datasets.brutal_dock.logger_utils import create_logging_tags
from LambdaZero.datasets.brutal_dock.mlflow_logger import MLFlowLogger
from LambdaZero.datasets.brutal_dock.model_trainer import MoleculeModelTrainer
from LambdaZero.datasets.brutal_dock.models import MessagePassingNet

torch.manual_seed(0)

path_of_this_file = Path(__file__).resolve()

experiment_name = 'First MPNN run'
tracking_uri = str(ROOT_DIR.joinpath("mlruns"))

num_epochs = 10
batch_size = 128
train_fraction = 0.8
validation_fraction = 0.1

best_model_path = RESULTS_DIR.joinpath("plumbing_tests")
d4_feather_data_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/dock_blocks105_walk40_clust.feather")
raw_data_path = RAW_EXPERIMENT_DATA_DIR.joinpath("dock_blocks105_walk40_clust.feather")

loss_function = F.mse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging_tags = create_logging_tags(execution_file_name=path_of_this_file)
mlflow_logger = MLFlowLogger(experiment_name, tracking_uri, logging_tags)

if __name__ == "__main__":
    logging.info(f"Executing {path_of_this_file}...")

    if not raw_data_path.is_file():
        logging.info(f"Copying {d4_feather_data_path} to {raw_data_path})")
        shutil.copy(str(d4_feather_data_path), str(raw_data_path))

    logging.info(f"Creating the full dataset")
    full_dataset = D4MoleculesDataset(str(EXPERIMENT_DATA_DIR))

    dataset_size = len(full_dataset)
    train_size = int(train_fraction * dataset_size)
    valid_size = int(validation_fraction * dataset_size)
    test_size = dataset_size - train_size - valid_size

    logging.info(f"Splitting data into train, validation, test sets")
    training_dataset, validation_dataset, _ = \
        torch.utils.data.random_split(full_dataset, [train_size, valid_size, test_size])

    logging.info(f"Creating dataloaders")
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    logging.info(f"Extracting mean and standard deviation from training data")
    training_mean, training_std = get_scores_statistics(training_dataloader)

    logging.info(f"Instantiating trainer and model")
    model_trainer = MoleculeModelTrainer(loss_function, device, mlflow_logger,
                                         score_mean=training_mean, score_std=training_std)

    model = MessagePassingNet()

    logging.info(f"Starting Model Training...")
    best_validation_loss = model_trainer.train_model(model,
                                                     training_dataloader,
                                                     validation_dataloader,
                                                     best_model_path,
                                                     num_epochs=num_epochs)

    logging.info(f"Best validation loss: {best_validation_loss: 5f}")
    mlflow_logger.log_metrics("best_val_loss", best_validation_loss)

    list_actuals, list_predicted = model_trainer.apply_model(model, validation_dataloader)

    list_errors = list_actuals-list_predicted

    mean_error = np.mean(list_errors)
    std_error = np.std(list_errors)

    info = f"Validation Results [real scale]: " \
           f"mean validation values : {np.mean(list_actuals):5f}, " \
           f"std on validation values : {np.std(list_actuals):5f}, " \
           f"mean error : {mean_error:5f}, " \
           f"std error : {std_error:5f}."
    logging.info(info)

    mlflow_logger.log_metrics("mean_validation_error_real_scale", mean_error)
    mlflow_logger.log_metrics("std_validation_error_real_scale", std_error)

    mlflow_logger.finalize()
