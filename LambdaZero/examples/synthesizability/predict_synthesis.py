'''
from LambdaZero.examples.synthesizability import ChempropPredictor # or predict_synthesis
synthesizability = ChempropPredictor(dataset = "/path/")
synthesizability.predict(smi = "ccc", binary=True/False) # or mol = mol
'''
import torch
import numpy as np
import sys, os
from rdkit import Chem
from collections import OrderedDict

from LambdaZero.utils import get_external_dirs
from chemprop.data import MoleculeDataLoader, MoleculeDataset, MoleculeDatapoint
from chemprop.utils import load_checkpoint, load_scalers
from chemprop.train import predict


class ChempropPredictor():
    def __init__(self, dataset=None, load_weights=True):
        self.no_cuda: bool = False
        self.dataset = dataset
        self.load_weights = load_weights
        self.model = None
        if self.load_weights:
            sys.stdout = open(os.devnull, "w")  # silience the checkpoint logger
            self.model = load_checkpoint(dataset, device=self.device)
            sys.stdout = sys.__stdout__
        self.scaler, self.features_scaler = load_scalers(dataset)


    @property
    def device(self) -> torch.device:
        if not self.cuda:
            return torch.device('cpu')

        return torch.device('cuda', self.gpu)

    @device.setter
    def device(self, device: torch.device) -> None:
        self.cuda = device.type == 'cuda'
        self.gpu = device.index

    @property
    def cuda(self) -> bool:
        return not self.no_cuda and torch.cuda.is_available()

    @cuda.setter
    def cuda(self, cuda: bool) -> None:
        self.no_cuda = not cuda

    def train_epoch(self, dataset):
        return {"loss"}

    def eval_epoch(self):
        return None

    def predict(self, smi=None, mol=None, binary=True):
        if smi is None:
            smi = Chem.MolToSmiles(mol)

        test_data = MoleculeDataset([
            MoleculeDatapoint(
                smiles=smi,
                row=OrderedDict({'smiles': smi}),
                features_generator=None
            )
        ])

        test_data_loader = MoleculeDataLoader(
            dataset=test_data,
        )

        model_preds = predict(
            model=self.model,
            data_loader=test_data_loader,
            disable_progress_bar=True,
            scaler=self.scaler
        ) # prediction = self.model(mol)

        model_preds = np.array(model_preds)[0][0]
        if binary:
            return int(round(model_preds))  # binary, you can also return fulldata.row for the smiles
        else:  # regression
            return model_preds


datasets_dir, programs_dir, summaries_dir = get_external_dirs()

if __name__ == '__main__':
    synthesizability = ChempropPredictor(dataset = os.path.join(datasets_dir, "synthesizability/binary_corrected/model_0/model.pt"))
    print (synthesizability.predict(smi = "Clc1cc(N2CCN(CC2)CCCN2c3c(c(OC)ccc3)CCC2=O)ccc1"))


#DEFAULT_CONFIG = {
#    "chemprop_pars"={num_layer=2, num_ensembles=4}}
#    "num_workers": 8  # Number of workers for the parallel data loading (0 means sequential)
#    "batch_size": int = 50  # Batch size
#    "no_features_scaling": = False
#}