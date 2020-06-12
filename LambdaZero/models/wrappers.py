import torch
from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph, MolGraph
from chemprop.train import cross_validate
from chemprop.utils import load_checkpoint, load_scalers, create_logger


class ChempropWrapper_v1:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # prediction
        self.checkpoint_path = config["predict_config"]["checkpoint_path"]
        self.predict_config = config["predict_config"]

        # training
        self.train_dataset = config["trainer_config"]["train_dataset"]
        self.train_save_dir = config["trainer_config"]["save_dir"]
        self.dataset_type = config["trainer_config"]["dataset_type"]
        self.trainer_config = config["trainer_config"]

        # to add features_generator
        if self.predict_config["checkpoint_path"] is not None:  # for prediction
            self.model = load_checkpoint(self.checkpoint_path, device=self.device)
            self.scaler, self.features_scaler = load_scalers(self.checkpoint_path)

    def train(self):
        args = TrainArgs()
        args.data_path = self.train_dataset
        args.dataset_type = self.dataset_type
        args.save_dir = self.train_save_dir
        for key, value in self.trainer_config:
            args.key = value
        logger = create_logger(name="train", save_dir=args.save_dir, quiet=True)
        mean_score, std_score = cross_validate(args, logger)
        return mean_score

    # to write a train_epoch, and hyperparameter_opt
    def __call__(self, mol):  # takes both mol and smiles

        mol = BatchMolGraph([MolGraph(mol)])

        model = self.model
        model.eval()
        features = self.predict_config["features_generator"]
        with torch.no_grad():
            preds = model(mol, features)
        preds = preds.data.cpu().numpy()

        # Inverse scale if regression
        if self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)
        return preds[0][0]
