'''
from LambdaZero.examples.synthesizability import predict_synthesizability
predict_synthesizability(path, smiles, "binary"/"regression") #to predict synthesizability
'''
from tap import Tap
import torch
from typing import List, Union
import numpy as np
import sys, os

from chemprop.args import TrainArgs
from chemprop.data import MoleculeDataLoader, MoleculeDataset
from chemprop.data.utils import get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.train import predict

class PredictArgs(Tap):
    checkpoint_paths: List[str]
    num_workers: int = 8   # Number of workers for the parallel data loading (0 means sequential)
    batch_size: int = 50  # Batch size
    no_features_scaling: bool = False

    @property
    def ensemble_size(self) -> int:
        return len(self.checkpoint_paths)

    def process_args(self) -> None:
        super(PredictArgs, self).process_args()

        if self.checkpoint_paths is None or len(self.checkpoint_paths) == 0:
            raise ValueError('Found no checkpoints. Must specify --checkpoint_path <path> or '
                             '--checkpoint_dir <dir> containing at least one checkpoint.')

    @property
    def features_scaling(self) -> bool:
        return not self.no_features_scaling

    @property
    def device(self) -> torch.device:
        if not self.cuda:
            print (1)
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


def make_predictions(args: PredictArgs, smiles: List[str] = None, type: str = "Binary") -> int: #-> List[Optional[List[float]]]:

    #print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])
    num_tasks, task_names = train_args.num_tasks, train_args.task_names

    # Update predict args with training arguments to create a merged args object
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    args: Union[PredictArgs, TrainArgs]

    #print('Loading data')
    full_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False, features_generator=args.features_generator)

    #print('Validating SMILES')
    full_to_valid_indices = {}
    valid_index = 0
    for full_index in range(len(full_data)):
        if full_data[full_index].mol is not None:
            full_to_valid_indices[full_index] = valid_index
            valid_index += 1

    test_data = MoleculeDataset([full_data[i] for i in sorted(full_to_valid_indices.keys())])

    avg_preds = np.zeros((len(test_data), num_tasks))

    # Create data loader
    test_data_loader = MoleculeDataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    #print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    # Load model
    sys.stdout = open(os.devnull, "w") #silience the checkpoint logger
    model = load_checkpoint(args.checkpoint_paths[0], device=args.device)
    sys.stdout = sys.__stdout__

    model_preds = predict(
        model=model,
        data_loader=test_data_loader,
        disable_progress_bar=True,
        scaler=scaler
    )
    avg_preds += np.array(model_preds)

    if type == "binary":
        return int(round(avg_preds[0][0])) #binary, you can also return fulldata.row for the smiles
    else: #default is regression
        return avg_preds[0][0]


def predict_synthesizability(checkpoint_path, smiles, type):
    args = PredictArgs()
    args.checkpoint_paths = [checkpoint_path]
    return make_predictions(args, [smiles], type)
