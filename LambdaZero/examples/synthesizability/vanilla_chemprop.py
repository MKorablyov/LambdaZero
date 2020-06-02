"""
prediction:
from LambdaZero.examples.synthesizability import ChempropPredictor, predict_config # or predict_synthesis
synthesizability = ChempropPredictor(checkpoint_path = "/path/")
synthesizability.predict(smi = "ccc", predict_config=predict_config) # or mol = mol, either one smiles string or a list

training:
from LambdaZero.examples.synthesizability import ChempropPredictor, trainer_config # or predict_synthesis
synthesizability = ChempropPredictor(molecule_dataset = "/path/", dataset_type = 'regression' or 'classification', save_dir = "/path/")
synthesizability.train(trainer_config=trainer_config) #or modify trainer_config
"""

import torch
import sys
import os
import numpy as np
from rdkit import Chem
from collections import OrderedDict

from LambdaZero.utils import get_external_dirs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
from chemprop.utils import load_checkpoint, load_scalers, create_logger
from chemprop.args import TrainArgs
from chemprop.train import predict
from chemprop.train import cross_validate


class ChempropModel:
    def __init__(self, checkpoint_path="kkk", dataset_type='classification', dataset=None, save_dir=None):
        #self.no_cuda: bool = False

        if checkpoint_path is None:
            pass

        self.gpu = None
        self.checkpoint_path = checkpoint_path
        self.load_weights = load_weights
        self.dataset_type = dataset_type  # regression, classification, (multiclass to be done)
        self.model = None

        self.molecule_dataset = molecule_dataset  # for training
        self.save_dir = save_dir
        if molecule_dataset is not None:
            self.load_weights = False
        # to add features_generator

        if self.load_weights:  #for prediction
            sys.stdout = open(os.devnull, "w")  # silience the checkpoint logger
            self.model = load_checkpoint(self.checkpoint_path, device=self.device)
            sys.stdout = sys.__stdout__
            self.scaler, self.features_scaler = load_scalers(self.checkpoint_path)

    # @property
    # def device(self) -> torch.device:
    #     if not self.cuda:
    #         return torch.device('cpu')
    #
    #     return torch.device('cuda', self.gpu)
    #
    # @device.setter
    # def device(self, device: torch.device) -> None:
    #     self.cuda = device.type == 'cuda'
    #     self.gpu = device.index
    #
    # @property
    # def cuda(self) -> bool:
    #     return not self.no_cuda and torch.cuda.is_available()
    #
    # @cuda.setter
    # def cuda(self, cuda: bool) -> None:
    #     self.no_cuda = not cuda

    def train(self):
        #args = TrainArgs().parse_args()                                         # todo: no parse args anywhere
        #args.data_path = self.molecule_dataset
        #args.dataset_type = self.dataset_type
        #args.save_dir = self.save_dir

        for key, value in trainer_config:
            args.key = value
        logger = create_logger(name='train', save_dir=args.save_dir, quiet=True)
        mean_score, std_score = cross_validate(args, logger)
        return mean_score

    # to write a train_epoch, and hyperparameter_opt

    def __call__(self, smi=None): #smi and mol can be a single string, or a list of strings


        #if smi is None:
        #    if isinstance(mol, list):
        #        smi = (Chem.MolFromSmiles(single_mol) for single_mol in mol)
        #    else:
        #        smi = Chem.MolToSmiles(mol)

        # if isinstance(smi, list): #if it's a list
        #     test_data = MoleculeDataset([
        #         MoleculeDatapoint(
        #             smiles=smile,
        #             row=OrderedDict({'smiles': smile}),
        #             features_generator=predict_config["features_generator"]
        #         ) for smile in smi
        #     ])
        # else:

        dataset = MoleculeDataset([
            MoleculeDatapoint(
                    smiles=str(smi),
                    row=OrderedDict({'smiles': smi}),
                    features_generator=predict_config["features_generator"]
                )
            ])

        data_loader = MoleculeDataLoader(
            dataset=dataset,
            batch_size=predict_config["batch_size"],
            num_workers=predict_config["num_workers"])

        preds = predict(
            model=self.model,
            data_loader=data_loader,
            disable_progress_bar=predict_config["disable_progress_bar"],
            scaler=self.scaler)

        return preds[0]

        # prediction = self.model(mol)
        #model_preds = np.array(model_preds)
        #if self.dataset_type == ("classification" or "multiclass"):
        #    # binary, you can also return fulldata.row for the smiles
        #    if isinstance(smi, list):
        #        return np.round(model_preds.astype(int), 0)  # return the list
        #    else:
        #        return np.round(model_preds.astype(int), 0)[0][0]
        # else:
        #     # regression
        #     if isinstance(smi, list):
        #         return model_preds
        #     else:
        #         return model_preds[0][0]


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

trainer_config = DEFAULT_CONFIG["trainer_config"]
predict_config = DEFAULT_CONFIG["predict_config"]
config = DEFAULT_CONFIG

if __name__ == '__main__':
    synthesizability = ChempropPredictor(dataset_type="classification",
                                         checkpoint_path=os.path.join(datasets_dir,
                                                                      "synthesizability/binary_corrected/model_0/model.pt"))
    print (synthesizability.predict(predict_config=predict_config,
                                    smi="Clc1cc(N2CCN(CC2)CCCN2c3c(c(OC)ccc3)CCC2=O)ccc1"))

