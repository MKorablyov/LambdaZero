from argparse import Namespace
from typing import List

import numpy
import torch
import os.path as osp
from torch_geometric.data import Data, Batch

from LambdaZero.examples.models import get_model
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments.molMDP import BlockMoleculeData

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class BaseProxy(torch.nn.Module):
    def __init__(self, args: Namespace):
        super(BaseProxy, self).__init__()

        self.args = args
        cuda = getattr(args, "cuda", True)
        self.device = device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        repr_args = args.representation
        model_args = args.model
        train_args = getattr(args, "train", None)

        # -- Initialize MolMDP representation variables & classes
        repr_args.device = device
        self._load_representation(repr_args)

        # -- Load torch.nn Model
        self.proxy_net = get_model(model_args, mdp=getattr(self, "mdp", None))
        self.proxy_net.to(device)

        # -- Load training stuff
        self._load_train(train_args)

    def __call__(self, mols: List[BlockMoleculeData]) -> numpy.ndarray:
        """ MUST exist - evaluate BlockMoleculeData and return true scores (no normalization) """
        mols = mols if isinstance(mols, list) else [mols]

        with torch.no_grad():
            batch = self.geom_data_2_batch([self.block_mol_2_repr(m) for m in mols])

            # Transform batch
            proxy_scores = self.proxy_predict(batch).data.cpu().numpy()
            scores = self.transform_predicted(proxy_scores)
        return scores

    def proxy_predict(self, batch: Batch) -> torch.Tensor:
        predictions = self.proxy_net(batch)
        return predictions

    def block_mol_2_repr(self, mol: BlockMoleculeData) -> Data:
        """ Receives a BlockMoleculeData point and returns torch.geometric.Data """
        return self.mdp.mol2repr(mol)

    def geom_data_2_batch(self, data_list: List[Data]) -> Batch:
        """ Generates a torch geometric Batch """
        return self.mdp.mols2batch(data_list)

    def transform_target(self, target_scores: numpy.ndarray) -> numpy.ndarray:
        """ Returns target values for the proxy model - from dockscores (e.g. normalize) """
        x_norm = (target_scores - self.target_norm[0]) / self.target_norm[1]
        return x_norm

    def transform_predicted(self, predicted_scores: numpy.ndarray) -> numpy.ndarray:
        """ Return true scores from predicted (e.g. denormalize) """
        scores = (predicted_scores * self.target_norm[1]) + self.target_norm[0]
        return scores

    def _load_train(self, args: Namespace):
        pass

    def _load_representation(self, args: Namespace) -> None:
        bpath = getattr(args, "bpath", osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"))

        self.mdp = MolMDPExtended(bpath)  # Used for generating representation
        mdp_init = getattr(args, "mdp_init", {"repr_type": "atom_graph"})
        mdp_init = getattr(mdp_init, "__dict__", mdp_init)
        mdp_init["device"] = args.device

        self.mdp.post_init(**mdp_init)

        self.target_norm = getattr(args, "target_norm", [-8.6, 1.10])

    def reset(self):
        """ Reset model net parameters """
        for layer in self.proxy_net.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def train(self, *args, **kwargs) -> dict:
        """ Train model from scratch. Return train metrics """
        raise NotImplementedError

    def fine_tune(self, *args, **kwargs):
        """ Fine-tune model with new data. Return finetune metrics """
        raise NotImplementedError

    def save(self, out_dir: str, score: float, info: dict = None):
        save_data = dict({
            "state_dict": self.proxy_net.state_dict(),
            "config": self.args,
            "info": info,
            "score": score
        })
        # TODO implement save best model vs last model based on score
        torch.save(save_data, f"{out_dir}/best_model.pk")

    def load(self, load_args: Namespace, checkpoint: dict) -> None:
        self.proxy_net.load_state_dict(checkpoint["state_dict"])
