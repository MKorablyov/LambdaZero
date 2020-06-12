#!/usr/bin/env python
"""
This experiment script applies the OptimizedChemprop model, which uses MolGraph instead of smiles,
to the D4 docking dataset.

to execute this experiment, invoke the script as:

    python optimized_chemprop_experiment.py --config=[INPUT CONFIGURATION]
                                            --working_directory=[WORKING_DIRECTORY]
                                            --output_directory=[OUTPUT_DIRECTORY]
                                            --data_directory=[DATA_DIRECTORY]
                                            --tracking_uri=[TRACKING_URI]
"""
import logging
from pathlib import Path

from orion.client import report_results

from LambdaZero.oracle_models.chemprop_model import MolGraphChempropNet
from LambdaZero.chemprop_adaptors.dataloader_utils import get_chemprop_dataloaders


# Specify which model class we want to instantiate and train
from LambdaZero.chemprop_adaptors import ChempropModelTrainer
from LambdaZero.representation_learning.datasets import D4ChempropMoleculesDataset
from LambdaZero.representation_learning.experiment_driver import experiment_driver
from LambdaZero.representation_learning.parameter_inputs import get_input_and_run_configuration

model_class = MolGraphChempropNet

# Specify which dataset class we want to use
dataset_class = D4ChempropMoleculesDataset

path_of_this_file = Path(__file__).resolve()

if __name__ == "__main__":
    logging.info(f"Executing {path_of_this_file}...")

    input_and_run_config = get_input_and_run_configuration(path_of_this_file)
    best_validation_loss = experiment_driver(input_and_run_config, dataset_class, model_class,
                                             get_dataloaders=get_chemprop_dataloaders,
                                             model_trainer_class=ChempropModelTrainer)

    report_results([dict(name="best_validation_loss", type="objective", value=best_validation_loss)])

