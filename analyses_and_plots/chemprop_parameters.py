from pathlib import Path

from LambdaZero.examples.representation_learning_experiments import EXPERIMENT_DATA_DIR
from LambdaZero.representation_learning.parameter_inputs import CONFIG_KEY, NON_CONFIG_KEY, EXECUTION_FILENAME_KEY, \
    PATHS_KEY
from LambdaZero.utils import get_external_dirs
from analyses_and_plots import ANALYSIS_RESULTS_DIR

data_dir_string, _, _ = get_external_dirs()

config = {"run_parameters": {"experiment_name": "Chemprop",
                             "run_name": "default_parameters"
                             },
          "training": {"num_epochs": 1,
                       "num_workers": 1,
                       "batch_size": 256,
                       "learning_rate": 0.001,
                       "train_fraction": 0.8,
                       "validation_fraction": 0.1,
                       "patience": 5
                       },
          "model": {"name": "chemprop",
                    "depth": 3,
                    "ffn_num_layers": 2,
                    "ffn_hidden_size": 300
                    }
          }

paths = {"tracking_uri": "fake_traking_uri",
         "working_directory": EXPERIMENT_DATA_DIR,
         "output_directory": ANALYSIS_RESULTS_DIR,
         "data_directory": str(Path(data_dir_string).joinpath("/brutal_dock/d4/raw"))
         }

non_config_dict = {PATHS_KEY: paths,
                   EXECUTION_FILENAME_KEY: "dummy"}


input_and_run_config = {CONFIG_KEY: config,
                        NON_CONFIG_KEY: non_config_dict}
