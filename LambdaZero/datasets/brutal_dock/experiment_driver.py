import logging
from pathlib import Path
from typing import Dict, Any, Type

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from LambdaZero.datasets.brutal_dock import set_logging_directory
from LambdaZero.datasets.brutal_dock.dataset_utils import get_scores_statistics
from LambdaZero.datasets.brutal_dock.dataset_splitting import get_split_datasets
from LambdaZero.datasets.brutal_dock.datasets import MoleculesDatasetBase
from LambdaZero.datasets.brutal_dock.metrics_utils import get_prediction_statistics
from LambdaZero.datasets.brutal_dock.mlflow_logger import MLFlowLogger
from LambdaZero.datasets.brutal_dock.model_trainer import MoleculeModelTrainer
from LambdaZero.datasets.brutal_dock.models import ModelBase
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, MODEL_PARAMETERS_KEY, \
    TRAINING_PARAMETERS_KEY, write_configuration_file

loss_function = F.mse_loss


def experiment_driver(
    config: Dict[str, Any],
    dataset_class: Type[MoleculesDatasetBase],
    model_class: Type[ModelBase],
    random_seed: int = 0,
) -> float:
    """
    This method drives the execution of an experiment. It is responsible for
    training a model based on a given dataset, evaluate the trained model and
    log all relevant information.
    input:
        config: dictionary containing all the needed parameters
        dataset: the full instantiated dataset, which will be split within this driver
        model_class: class for the model, which should derive from ModelBase.
    """

    torch.manual_seed(random_seed)

    logging.info(f"Parsing input arguments")
    run_parameters = config.pop(RUN_PARAMETERS_KEY)
    training_parameters = config[TRAINING_PARAMETERS_KEY]
    model_parameters = config[MODEL_PARAMETERS_KEY]

    data_dir = Path(run_parameters.pop("data_directory"))
    work_dir = Path(run_parameters.pop("working_directory"))
    out_dir = Path(run_parameters.pop("output_directory"))

    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging_directory = out_dir.joinpath("logs/")
    logging_directory.mkdir(parents=True, exist_ok=True)
    set_logging_directory(logging_directory)

    logging.info(f"Instantiating full dataset")
    dataset = dataset_class.create_dataset(root_dir=work_dir,
                                           original_raw_data_dir=data_dir)

    logging.info(f"Instantiating model for training")
    model = model_class.create_model_for_training(model_parameters)

    logging.info(f"Splitting data into train, validation, and test sets")
    training_dataset, validation_dataset, test_dataset = \
        get_split_datasets(dataset,
                           training_parameters['train_fraction'],
                           training_parameters['validation_fraction'])

    logging.info(f"size of training set {len(training_dataset)}")
    logging.info(f"size of validation set {len(validation_dataset)}")
    logging.info(f"size of test set {len(test_dataset)}")

    logging.info(f"Creating dataloaders")
    training_dataloader = DataLoader(training_dataset,
                                     batch_size=training_parameters['batch_size'],
                                     num_workers=training_parameters['num_workers'],
                                     shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=training_parameters['batch_size'],
                                       num_workers=training_parameters['num_workers'],
                                       shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=training_parameters['batch_size'],
                                 num_workers=training_parameters['num_workers'],
                                 shuffle=False)

    logging.info(f"Extracting mean and standard deviation from training data")
    training_mean, training_std = get_scores_statistics(training_dataloader)

    logging.info(f"Instantiating model trainer")

    tracking_uri = Path(run_parameters.pop("tracking_uri"))
    tracking_uri.mkdir(parents=True, exist_ok=True)
    experiment_logger = MLFlowLogger(run_parameters.pop("experiment_name"),
                                     str(tracking_uri),
                                     run_parameters)
    experiment_logger.log_parameters("training", training_parameters)
    experiment_logger.log_parameters("model", model_parameters)

    logging.info(f"Writing configuration to artifact directory")
    json_config_path = str(out_dir.joinpath("config.json"))
    config = {TRAINING_PARAMETERS_KEY: training_parameters,
              MODEL_PARAMETERS_KEY: model_parameters}

    write_configuration_file(json_config_path, config)
    experiment_logger.log_artifact(json_config_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_trainer = MoleculeModelTrainer(loss_function,
                                         device,
                                         experiment_logger,
                                         score_mean=training_mean,
                                         score_std=training_std)

    logging.info(f"Starting Model Training...")
    best_model_path = out_dir.joinpath("best_model.ckpt")

    best_validation_loss = model_trainer.train_model(model,
                                                     training_dataloader,
                                                     validation_dataloader,
                                                     best_model_path,
                                                     lr=training_parameters["learning_rate"],
                                                     num_epochs=training_parameters["num_epochs"])

    logging.info(f"Best validation loss: {best_validation_loss: 5f}")
    experiment_logger.increment_step_and_log_metrics("best_val_loss", best_validation_loss)

    logging.info(f"Instantiate best model observed in training, loading from {best_model_path}")
    best_model = model_class.load_model_from_file(best_model_path, model_parameters)

    for label, dataloader in zip(['validation', 'test'], [validation_dataloader, test_dataloader]):
        list_actuals, list_predicted = model_trainer.apply_model(best_model, dataloader)
        mean_absolute_error, std_absolute_error = get_prediction_statistics(list_actuals, list_predicted)
        mean_key = f"{label}_mean_absolute_error_real_scale"
        std_key = f"{label}_std_absolute_error_real_scale"
        experiment_logger.increment_step_and_log_metrics(mean_key, mean_absolute_error)
        experiment_logger.increment_step_and_log_metrics(std_key, std_absolute_error)

    logging.info(f"Finalizing.")
    experiment_logger.finalize()

    return best_validation_loss

