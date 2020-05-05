"""
This experiment script applies a Message Passing Neural Net model
to the D4 docking dataset.

It assumes the D4 docking data is available in a feather file.
"""
import logging
from pathlib import Path

from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.models import MessagePassingNet
from LambdaZero.datasets.brutal_dock.parameter_inputs import get_input_and_run_configuration

# Specify which model class we want to instantiate and train
model_class = MessagePassingNet

# Specify which dataset class we want to use
dataset_class = D4MoleculesDataset

path_of_this_file = Path(__file__).resolve()

if __name__ == "__main__":
    logging.info(f"Executing {path_of_this_file}...")

    input_and_run_config = get_input_and_run_configuration(path_of_this_file)
    best_validation_loss = experiment_driver(input_and_run_config, dataset_class, model_class)
