"""
prediction:
from LambdaZero.examples.synthesizability import ChempropModel, config
synthesizability = ChempropModel(config=config, dataset_type = "classification", checkpoint_path = "/path/")
synthesizability(mol = "ccc") # or mol from rdkit.Chem

training:
from LambdaZero.examples.synthesizability import ChempropModel, config
synthesizability = ChempropModel(config=config, dataset_type = "classification", train_dataset = "/path/", train_save_dir = "/path/")
synthesizability.train()
"""

import os
from rdkit import Chem
from copy import deepcopy
from LambdaZero.utils import get_external_dirs
from LambdaZero.models import ChempropWrapper_v1



datasets_dir, programs_dir, summaries_dir = get_external_dirs()

DEFAULT_CONFIG = {
    "trainer_config": {
        "dataset_type": "regression",
        "train_dataset": None,
        "save_dir": None,

        "features_generator": None,  # Method(s) of generating additional features
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the acquire data loading (0 means sequential)
        "batch_size": 50,  # Batch size
        "separate_val_path": None,  # Path to separate val set, optional
        "separate_test_path": None,  # Path to separate test set, optional
        "split_type": "random",
        "split_sizes": (0.8, 0.1, 0.1),
        "num_folds": 1,
        "seed": 0,
        "pytorch_seed": 0,
        "log_frequency": 10,
        "cache_cutoff": 10000,
        "save_smiles_splits": False,

        "hidden_size": 300,
        "depth": 3,
        "dropout": 0.0,
        "activation": "ReLu",
        "ffn_num_layers": 2,
        "ensemble_size": 1,
        "atom_messages": False,  # Centers messages on atoms instead of on bonds
        "undirected": False,

        "epochs": 150,
        "warmup_epochs": 2.0,  # epochs for which lr increases linearly; afterwards decreases exponentially
        "init_lr": 1e-4,  # Initial learning rate
        "max_lr": 1e-3,  # Maximum learning rate
        "final_lr":  1e-4,  # Final learning rate
        "class_balance": False,
        },
    "predict_config": {
        "dataset_type": "regression",
        "features_generator": None,
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the acquire data loading (0 means sequential)
        "batch_size": 50,  # Batch size
        "disable_progress_bar": True,
    },
}

synth_config = deepcopy(DEFAULT_CONFIG)
synth_config["predict_config"]["checkpoint_path"] = os.path.join(datasets_dir,
                                            "Synthesizability/MPNN_model/Regression/model_2/model.pt")
synth_config["synth_cutoff"] = [0, 4]

binding_config = deepcopy(DEFAULT_CONFIG)
binding_config["predict_config"]["checkpoint_path"] = os.path.join(datasets_dir,
                                            "brutal_dock/mpro_6lze/trained_weights/chemprop/model_0/model.pt")
binding_config["dockscore_std"] = [-49.411, 7.057]


if __name__ == '__main__':

    synthesizability = ChempropWrapper_v1(config=synth_config)
    mol = Chem.MolFromSmiles("Clc1cc(N2CCN(CC2)CCCN2c3c(c(OC)ccc3)CCC2=O)ccc1")
    # 6 is average for zinc
    print(synthesizability(mol=mol))
    # /home/maksym/Datasets/Synthesizability/MPNN_model/Binary_corrected/model_0/model.pt
    # /home/maksym/Datasets/Synthesizability/binary_corrected/model_0/model.pt
