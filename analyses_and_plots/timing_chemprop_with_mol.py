"""
The goal of this analysis script is to identify the execution bottleneck in the chemprop model so
we can focus on the correct thing to improve to accelerate execution.
"""
from LambdaZero.datasets.brutal_dock.analysis.chemprop_parameters import input_and_run_config
from LambdaZero.representation_learning.chemprop_adaptors.dataloader_utils import get_chemprop_dataloaders
from LambdaZero.representation_learning.chemprop_adaptors import ChempropModelTrainer
from LambdaZero.datasets.brutal_dock.datasets import D4ChempropMoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.loggers.null_logger import NullLogger
from LambdaZero.datasets.brutal_dock.models.chemprop_model import OptimizedChempropNet

dataset_class = D4ChempropMoleculesDataset
model_class = OptimizedChempropNet

if __name__ == '__main__':

    _ = experiment_driver(input_and_run_config, dataset_class, model_class,
                          logger_class=NullLogger,
                          get_dataloaders=get_chemprop_dataloaders,
                          model_trainer_class=ChempropModelTrainer)

