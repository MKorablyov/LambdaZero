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

import torch
import sys
import os
from rdkit import Chem

from LambdaZero.utils import get_external_dirs
from chemprop.utils import load_checkpoint, load_scalers, create_logger
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.args import TrainArgs
from chemprop.train import cross_validate


class ChempropModel:
    def __init__(self, config, checkpoint_path=None, dataset_type='classification', train_dataset=None, train_save_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        load_weights = True
        if checkpoint_path is None:
            load_weights = False
        if train_dataset is not None:
            load_weights = False

        # prediction
        self.checkpoint_path = checkpoint_path
        self.load_weights = load_weights
        self.dataset_type = dataset_type  # regression, classification, (multiclass to be done)
        self.predict_config = config["predict_config"]
        self.model = None

        # training
        self.train_dataset = train_dataset
        self.train_save_dir = train_save_dir
        self.trainer_config = config["trainer_config"]

        # to add features_generator
        if load_weights:  #for prediction
            sys.stdout = open(os.devnull, "w")  # silience the checkpoint logger
            self.model = load_checkpoint(self.checkpoint_path, device=self.device)
            sys.stdout = sys.__stdout__
            self.scaler, self.features_scaler = load_scalers(self.checkpoint_path)

    def train(self):
        args = TrainArgs()
        args.data_path = self.train_dataset
        args.dataset_type = self.dataset_type
        args.save_dir = self.train_save_dir
        for key, value in self.trainer_config:
            args.key = value
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=True)
        mean_score, std_score = cross_validate(args, logger)
        return mean_score

    # to write a train_epoch, and hyperparameter_opt
    def __call__(self, mol=None): #takes both mol and smiles

        mol = BatchMolGraph([MolGraph(mol)])

        model = self.model
        model.eval()
        features = self.predict_config["features_generator"]
        with torch.no_grad(): preds = model(mol, features)
        preds = preds.data.cpu().numpy()

        # Inverse scale if regression
        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)
        return preds[0][0]


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

DEFAULT_CONFIG = {
    "trainer_config": {
        "features_generator": None,  # Method(s) of generating additional features
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the parallel data loading (0 means sequential)
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
        "features_generator": None,
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the parallel data loading (0 means sequential)
        "batch_size": 50,  # Batch size
        "disable_progress_bar": True,
    },
}

config = DEFAULT_CONFIG

if __name__ == '__main__':

    synthesizability = ChempropModel(config=config, dataset_type="classification",
                                     checkpoint_path=os.path.join(datasets_dir,
                                     "Synthesizability/MPNN_model/Binary_corrected/model_0/model.pt"))
    mol = Chem.MolFromSmiles("Clc1cc(N2CCN(CC2)CCCN2c3c(c(OC)ccc3)CCC2=O)ccc1")
    print (synthesizability(mol=mol))

    # /home/maksym/Datasets/Synthesizability/MPNN_model/Binary_corrected/model_0/model.pt

    # /home/maksym/Datasets/Synthesizability/binary_corrected/model_0/model.pt