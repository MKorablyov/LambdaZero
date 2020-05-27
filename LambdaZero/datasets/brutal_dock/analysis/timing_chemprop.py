"""
The goal of this analysis script is to identify the execution bottleneck in the chemprop model so
we can focus on the correct thing to improve to accelerate execution.
"""
from pathlib import Path


from LambdaZero.datasets.brutal_dock import RESULTS_DIR, BRUTAL_DOCK_DATA_DIR
from LambdaZero.datasets.brutal_dock.datasets import D4GeometricMoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.experiments import EXPERIMENT_DATA_DIR
from LambdaZero.datasets.brutal_dock.loggers.null_logger import NullLogger
from LambdaZero.datasets.brutal_dock.models.chemprop_model import ChempropNet
from LambdaZero.datasets.brutal_dock.parameter_inputs import CONFIG_KEY, NON_CONFIG_KEY, \
    PATHS_KEY, EXECUTION_FILENAME_KEY

config = {"run_parameters": {
    "experiment_name": "Chemprop",
    "run_name": "default_parameters"
    },
    "training": {
        "num_epochs": 1,
        "num_workers": 1,
        "batch_size": 256,
        "learning_rate": 0.001,
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "patience": 5
    },
    "model": {
        "name": "chemprop",
        "depth": 3,
        "ffn_num_layers": 2,
        "ffn_hidden_size": 300
    }
}

paths = {"tracking_uri": "fake_traking_uri",
         "working_directory": EXPERIMENT_DATA_DIR,
         "output_directory": RESULTS_DIR,
         "data_directory": BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/"),
         }

non_config_dict = {PATHS_KEY: paths,
                   EXECUTION_FILENAME_KEY: "dummy"}


input_and_run_config = {CONFIG_KEY: config,
                        NON_CONFIG_KEY: non_config_dict}

dataset_class = D4GeometricMoleculesDataset
model_class = ChempropNet

if __name__ == '__main__':

    _ = experiment_driver(input_and_run_config, dataset_class, model_class, logger_class=NullLogger)
