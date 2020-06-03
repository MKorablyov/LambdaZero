import os,sys, time
import torch as th
from chemprop.utils import load_checkpoint, load_scalers, create_logger
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.args import TrainArgs
from chemprop.train import cross_validate


class ChempropWrapper_v1:
    def __init__(self, config):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

        checkpoint_path = None
        dataset_type = 'classification'
        train_dataset = None,
        train_save_dir = None

        # load_weights = True
        # if checkpoint_path is None:
        #     load_weights = False
        # if train_dataset is not None:
        #     load_weights = False

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
        if config['laod weights'] is not None:  # for prediction
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
    def __call__(self, mol=None):  # takes both mol and smiles

        mol = BatchMolGraph([MolGraph(mol)])

        model = self.model
        model.eval()
        features = self.predict_config["features_generator"]
        with th.no_grad(): preds = model(mol, features)
        preds = preds.data.cpu().numpy()

        # Inverse scale if regression
        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)
        return preds[0][0]