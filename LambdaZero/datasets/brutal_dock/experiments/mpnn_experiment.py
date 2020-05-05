"""
This experiment script applies a Message Passing Neural Net model
to the D4 docking dataset.

It assumes the D4 docking data is available in a feather file.
"""
import logging
from pathlib import Path

import torch

from LambdaZero.datasets.brutal_dock import ROOT_DIR, RESULTS_DIR, BRUTAL_DOCK_DATA_DIR
from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.experiments import EXPERIMENT_DATA_DIR, RAW_EXPERIMENT_DATA_DIR
from LambdaZero.datasets.brutal_dock.models import MessagePassingNet
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, TRAINING_PARAMETERS_KEY, \
    MODEL_PARAMETERS_KEY, augment_configuration_with_run_parameters

# Specify which model class we want to instantiate and train
model_class = MessagePassingNet
dataset_class = D4MoleculesDataset

torch.manual_seed(0)


path_of_this_file = Path(__file__).resolve()

data_dir = BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/")
work_dir = EXPERIMENT_DATA_DIR


def get_config():
    """
    TODO: This method should be superseded by a mechanism to read these parameters from input and config file.
    """
    run_parameters = dict(tracking_uri=str(ROOT_DIR.joinpath("mlruns")),
                          experiment_name='Debugging MPNN runs',
                          run_name="hard-coded-parameters",
                          )

    training_parameters = dict(num_epochs=100,
                               num_workers=4,
                               batch_size=4096,
                               learning_rate=1e-4,
                               train_fraction=0.8,
                               validation_fraction=0.1)

    model_parameters = dict(name="MPNN",
                            gcn_size=128,
                            edge_hidden=128,
                            gru_out=128,
                            gru_layers=1,
                            linear_hidden=128)

    config = {RUN_PARAMETERS_KEY: run_parameters,
              TRAINING_PARAMETERS_KEY: training_parameters,
              MODEL_PARAMETERS_KEY: model_parameters}

    return config


if __name__ == "__main__":
    logging.info(f"Executing {path_of_this_file}...")

    input_config = get_config()
    augmented_config = augment_configuration_with_run_parameters(input_config,
                                                                 path_of_this_file,
                                                                 working_directory=work_dir,
                                                                 output_directory=RESULTS_DIR,
                                                                 data_directory=data_dir)

    best_validation_loss = experiment_driver(augmented_config, dataset_class, model_class)
