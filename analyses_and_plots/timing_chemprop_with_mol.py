"""
The goal of this analysis script is to identify the execution bottleneck in the chemprop model so
we can focus on the correct thing to improve to accelerate execution.
"""
from LambdaZero.loggers.null_logger import NullLogger
from LambdaZero.oracle_models.chemprop_model import MolGraphChempropNet
from LambdaZero.chemprop_adaptors.dataloader_utils import get_chemprop_dataloaders
from LambdaZero.chemprop_adaptors import ChempropModelTrainer
from LambdaZero.examples.chemprop.datasets import D4ChempropMoleculesDataset
from LambdaZero.representation_learning.experiment_driver import experiment_driver
from analyses_and_plots.chemprop_parameters import input_and_run_config

dataset_class = D4ChempropMoleculesDataset
model_class = MolGraphChempropNet

if __name__ == '__main__':

    _ = experiment_driver(input_and_run_config, dataset_class, model_class,
                          logger_class=NullLogger,
                          get_dataloaders=get_chemprop_dataloaders,
                          model_trainer_class=ChempropModelTrainer)

