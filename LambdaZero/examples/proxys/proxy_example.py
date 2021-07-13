from argparse import Namespace
from typing import List

import numpy
import torch
import os.path as osp
from torch_geometric.data import Data, Batch

from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.examples.models import get_model
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended
from LambdaZero.utils import get_external_dirs
from LambdaZero.environments.molMDP import BlockMoleculeData
import LambdaZero

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class ProxyExample(BaseProxy):
    def __init__(self, args: Namespace):
        args.representation.mdp_repr.add_stem_mask = False  # TODO fix from checkpoint
        super(ProxyExample, self).__init__(args)
        self._transform = LambdaZero.utils.Complete()

    def proxy_predict(self, batch: Batch) -> torch.Tensor:
        predictions, _ = self.proxy_net(batch)
        return predictions.flatten()

    def block_mol_2_repr(self, mol: BlockMoleculeData) -> Data:
        """ Receives a BlockMoleculeData point and returns torch.geometric.Data """
        data_repr = self.mdp.mol2repr(mol, **self.mdp_repr_kwargs)
        return self._transform(data_repr)
