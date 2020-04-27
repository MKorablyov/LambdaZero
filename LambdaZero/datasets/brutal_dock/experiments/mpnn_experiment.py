"""
TODO : description
"""
import logging
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from LambdaZero.datasets.brutal_dock import ROOT_DIR, RESULTS_DIR, BRUTAL_DOCK_DATA_DIR
from LambdaZero.datasets.brutal_dock.graph_score_normalizer import GraphScoreNormalizer
from LambdaZero.datasets.brutal_dock.logger_utils import create_logging_tags
from LambdaZero.datasets.brutal_dock.mlflow_logger import MLFlowLogger
from LambdaZero.datasets.brutal_dock.model_trainer import MoleculeModelTrainer
from LambdaZero.datasets.brutal_dock.models import MessagePassingNet
from LambdaZero.datasets.brutal_dock.dataset_utils import get_smiles_and_scores_from_feather, \
    get_molecule_graph_dataset, get_scores_statistics
from LambdaZero.datasets.model_evaluator import MoleculeModelEvaluator

torch.manual_seed(0)

path_of_this_file = Path(__file__).resolve()

experiment_name = 'First MPNN run'
tracking_uri = str(ROOT_DIR.joinpath("mlruns"))

num_epochs = 100
dataset_size = 47410
batch_size = 128

train_size = int(0.8*dataset_size)
valid_size = int(0.1*dataset_size)
test_size = dataset_size - train_size - valid_size

best_model_path = RESULTS_DIR.joinpath("plumbing_tests")
d4_feather_data_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/dock_blocks105_walk40_clust.feather")

loss_function = F.mse_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging_tags = create_logging_tags(execution_file_name=path_of_this_file)
mlflow_logger = MLFlowLogger(experiment_name, tracking_uri, logging_tags)

if __name__ == "__main__":
    logging.info(f"Executing {path_of_this_file}...")

    logging.info(f"Reading SMILES and dockscores from {d4_feather_data_path}")
    list_smiles, list_scores = get_smiles_and_scores_from_feather(d4_feather_data_path)

    logging.info(f"Converting SMILES to molecular graphs")
    list_graphs = get_molecule_graph_dataset(list_smiles[:dataset_size], list_scores[:dataset_size])

    logging.info(f"Splitting data into train, validation, test sets")
    training_graphs, validation_graphs, _ = \
        torch.utils.data.random_split(list_graphs, [train_size, valid_size, test_size])

    logging.info(f"Extracting mean and standard deviation from training data")
    training_mean, training_std = get_scores_statistics(training_graphs)
    normalizer = GraphScoreNormalizer(training_mean, training_std)

    logging.info(f"Normalizing data")
    normalized_training_graphs = [normalizer.normalize_score(g) for g in training_graphs]
    normalized_validation_graphs = [normalizer.normalize_score(g) for g in validation_graphs]

    logging.info(f"Creating dataloaders")
    training_dataloader = DataLoader(normalized_training_graphs, batch_size=batch_size, num_workers=0)
    validation_dataloader = DataLoader(normalized_validation_graphs, batch_size=batch_size, num_workers=0)

    logging.info(f"Instantiating trainer and model")
    model_trainer = MoleculeModelTrainer(loss_function, device, mlflow_logger)

    model = MessagePassingNet()

    logging.info(f"Starting Model Training...")
    best_validation_loss = model_trainer.train_model(model,
                                                     training_dataloader,
                                                     validation_dataloader,
                                                     best_model_path,
                                                     num_epochs=num_epochs)

    mlflow_logger.log_metrics("best_val_loss", best_validation_loss)

    model_evaluator = MoleculeModelEvaluator(device)

    list_normalized_actuals, list_normalized_predicted = model_evaluator.get_actuals_and_predicted(validation_dataloader, model)

    list_actuals = training_std*list_normalized_actuals + training_mean
    list_predicted = training_std*list_normalized_predicted + training_mean

    list_errors = list_actuals-list_predicted

    mean_error = np.mean(list_errors)
    std_error = np.std(list_errors)

    mlflow_logger.log_metrics("mean_validation_error_real_scale", mean_error)
    mlflow_logger.log_metrics("std_validation_error_real_scale", std_error)

    mlflow_logger.finalize()
