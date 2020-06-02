"""
The goal of this analysis script is to identify the execution bottleneck in the chemprop model so
we can focus on the correct thing to improve to accelerate execution.
"""

from LambdaZero.loggers.null_logger import NullLogger
from LambdaZero.representation_learning.datasets import D4GeometricMoleculesDataset
from LambdaZero.representation_learning.experiment_driver import experiment_driver
from LambdaZero.representation_learning.models.chemprop_model import ChempropNet
from analyses_and_plots.chemprop_parameters import input_and_run_config

dataset_class = D4GeometricMoleculesDataset
model_class = ChempropNet

if __name__ == '__main__':

    _ = experiment_driver(input_and_run_config, dataset_class, model_class, logger_class=NullLogger)
