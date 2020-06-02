"""
The goal of this analysis script is to identify the execution bottleneck in the chemprop model so
we can focus on the correct thing to improve to accelerate execution.
"""
from LambdaZero.datasets.brutal_dock.analysis.chemprop_parameters import input_and_run_config
from LambdaZero.datasets.brutal_dock.datasets import D4GeometricMoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.loggers.null_logger import NullLogger
from LambdaZero.datasets.brutal_dock.models.chemprop_model import ChempropNet

dataset_class = D4GeometricMoleculesDataset
model_class = ChempropNet

if __name__ == '__main__':

    _ = experiment_driver(input_and_run_config, dataset_class, model_class, logger_class=NullLogger)
