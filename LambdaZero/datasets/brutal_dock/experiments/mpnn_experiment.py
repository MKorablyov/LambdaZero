"""
This experiment script applies a Message Passing Neural Net model
to the D4 docking dataset.

It assumes the D4 docking data is available in a feather file.
"""
import logging
import shutil
from pathlib import Path

import torch

from LambdaZero.datasets.brutal_dock import ROOT_DIR, RESULTS_DIR, BRUTAL_DOCK_DATA_DIR
from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset
from LambdaZero.datasets.brutal_dock.experiment_driver import experiment_driver
from LambdaZero.datasets.brutal_dock.experiments import EXPERIMENT_DATA_DIR, RAW_EXPERIMENT_DATA_DIR
from LambdaZero.datasets.brutal_dock.models import MessagePassingNet
from LambdaZero.datasets.brutal_dock.parameter_inputs import RUN_PARAMETERS_KEY, TRAINING_PARAMETERS_KEY, \
    MODEL_PARAMETERS_KEY, augment_configuration_with_run_parameters

torch.manual_seed(0)


path_of_this_file = Path(__file__).resolve()

d4_feather_data_path = BRUTAL_DOCK_DATA_DIR.joinpath("d4/raw/dock_blocks105_walk40_clust.feather")
raw_data_path = RAW_EXPERIMENT_DATA_DIR.joinpath("dock_blocks105_walk40_clust.feather")


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
                                                                 RESULTS_DIR,
                                                                 RESULTS_DIR,
                                                                 RAW_EXPERIMENT_DATA_DIR)

    # TODO: this is very idiosyncratic to this specific dataset. Should be generalized,
    #  and maybe done within the driver itself
    if not raw_data_path.is_file():
        logging.info(f"Copying {d4_feather_data_path} to {raw_data_path})")
        shutil.copy(str(d4_feather_data_path), str(raw_data_path))

    logging.info(f"Creating the full dataset")
    dataset = D4MoleculesDataset(str(EXPERIMENT_DATA_DIR))

    logging.info(f"Instantiating the model")

    # TODO: instantiating the model should be done within the driver to insure the correct
    #  parameters are used
    model = MessagePassingNet()

    best_validation_loss = experiment_driver(augmented_config, dataset, model)
