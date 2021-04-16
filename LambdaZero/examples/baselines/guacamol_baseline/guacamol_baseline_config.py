import os.path as osp
import LambdaZero.utils
import LambdaZero.inputs
from LambdaZero.contrib.reward import ProxyRewardSparse
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.inputs import temp_load_data_v1
from LambdaZero.contrib.config_model import model_config
from LambdaZero.contrib.model_with_uncertainty import MolMCDropGNN

from LambdaZero.contrib.config_model import load_seen_config
from LambdaZero.contrib.config_acquirer import oracle_config, acquirer_config

from LambdaZero.examples.baselines.guacamol_baseline.generator_wrapper import SmilesRnnDirectedGenerator_wrapper,  ChemGEGenerator_wrapper
# from LambdaZero.examples.baselines.guacamol_baseline.smiles_lstm_hc.smiles_rnn_directed_generator import SmilesRnnDirectedGenerator
# from LambdaZero.examples.baselines.guacamol_baseline.smiles_ga.goal_directed_generation import ChemGEGenerator
from LambdaZero.examples.baselines.guacamol_baseline.scoring_functions import Oracle_wrapper, Proxy_wrapper

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


# proxy_config = {
#     "update_freq": 10000,
#     "acquirer_config":acquirer_config,
#     "oracle": DockingOracle,
#     "oracle_config":oracle_config,
#     "load_seen": temp_load_data_v1,
#     "load_seen_config": load_seen_config,
# }

default_tune_config = {
    "number_molecules": 128, # num to generate via optimizer
    "starting_population": [],

    "evaluator": Proxy_wrapper,
    "reward": ProxyRewardSparse,
    "reward_config": {
        "synth_cutoff": [0, 4], # no synth_cutoff
        "synth_options": {"num_gpus": 0.05},
        "qed_cutoff": [0.2, 0.5],
        "always_discount": True,
        "exp_dock": False,
        # "clip_dockreward": None, # originally 2.5
        "scoreProxy": ProxyUCB,
        "scoreProxy_config": {
            "update_freq": 1000,
            "acquirer_config": {
                "model": MolMCDropGNN,
                "model_config": model_config,
                "acq_size": 32,
                "kappa": 0.0
            }, # acquirer_config,
            "oracle": DockingOracle,
            "oracle_config": oracle_config,
            "load_seen": temp_load_data_v1,
            "load_seen_config":
                {
                    "mean": -8.6, "std": 1.1,
                    "raw_path": osp.join(datasets_dir, "brutal_dock/seh/raw"),
                    "proc_path": osp.join(datasets_dir, "brutal_dock/seh/processed_rlbo"),
                    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_30k_debug3k.npy"),
                    "file_names": ["Zinc20_docked_neg_randperm_30k"],
                }, #load_seen_config,
        }, # proxy_config,
        "scoreProxy_options": {"num_cpus":2, "num_gpus": 1.0},
        "actor_sync_freq": 500,
        "device":  "cuda", #"cpu", #
    },

    "logger_config": {
        "wandb": {
            "project": "rlbo",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }},
}


DEFAULT_CONFIG = {
    "tune_config": {
        "config": default_tune_config,
        "local_dir": summaries_dir,
        "run_or_experiment": SmilesRnnDirectedGenerator_wrapper,
        "checkpoint_freq": 250,
        "stop": {"training_iteration": 2000},
    },
    "memory": 30 * 10 ** 9,
    "object_store_memory": 30 * 10 ** 9
}

lstm_001 = {
    "tune_config":{
        "run_or_experiment": SmilesRnnDirectedGenerator_wrapper,
        "stop": {"training_iteration": 20}, # 20 epochs, step method
        "config":{
            "method_config": {
                "pretrained_model_path": osp.join(datasets_dir, 'guacamol_baselines/smiles_lstm_hc/pretrained_model', 'model_final_0.473.pt'),
                "n_epochs": 20,
                "mols_to_sample": 1024,
                "keep_top": 512,
                "optimize_n_epochs": 5,
                "max_len": 100, # Max length of a SMILES string
                "optimize_batch_size": 256,
                "number_final_samples": 4096,
                "random_start": True,
                "smi_file": osp.join(datasets_dir, 'guacamol_baselines/data/guacamol_v1_all.smiles'),
                "n_jobs": -1, # guacamol uses joblib.Parallel
                "sample_final_model_only": False,
            }
        }
    }
}

ga_001 = {
    "tune_config": {
        "run_or_experiment": ChemGEGenerator_wrapper,
        "stop": {"training_iteration": 1000},  # 20 generations, step method
        "config": {
            "method_config": {
                "population_size": 100,
                "n_mutations": 200,
                "gene_size": 300,
                "generations": 1000,
                "patience": 5,
                "random_start": True,
                "smi_file": osp.join(datasets_dir, 'guacamol_baselines/data/guacamol_v1_all.smiles'),
                "n_jobs": -1,  # guacamol uses joblib.Parallel
            }
        }
    }
}

# lstm_001 = {
#     "method": SmilesRnnDirectedGenerator,
#     "method_config":{
#         "pretrained_model_path": osp.join(datasets_dir, 'guacamol_baseline/smiles_lstm_hc/pretrained_model', 'model_final_0.473.pt'),
#         "n_epochs": 20,
#         "mols_to_sample": 1024,
#         "keep_top": 512,
#         "optimize_n_epochs": 5,
#         "max_len": 100, # Max length of a SMILES string
#         "optimize_batch_size": 256,
#         "number_final_samples": 4096,
#         "random_start": True,
#         "smi_file": osp.join(datasets_dir, 'guacamol_baseline/data/guacamol_v1_all.smiles'),
#         "n_jobs": -1, # guacamol uses joblib.Parallel
#     }
# }
#
# ga_001 = {
#     "method": ChemGEGenerator,
#     "method_config": {
#         "population_size": 100,
#         "n_mutations": 200,
#         "gene_size": 300,
#         "generations": 1000,
#         "patience": 5,
#         "random_start": True,
#         "smi_file": osp.join(datasets_dir, 'guacamol_baseline/data/guacamol_v1_all.smiles'),
#         "n_jobs": -1,  # guacamol uses joblib.Parallel
#     }
# }
