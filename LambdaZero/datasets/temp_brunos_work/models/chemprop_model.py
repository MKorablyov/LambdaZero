from chemprop.args import TrainArgs
from torch_geometric.data import Batch
import numpy as np

from chemprop.models import MoleculeModel

from LambdaZero.datasets.temp_brunos_work.models.model_base import ModelBase


class ChempropNet(ModelBase):
    """
    This model class adapts the chemprop model to fit in within LambdaZero's framework.
    The Forward method assumes that the pytorch_geometric batch of molecules contains
    a field "smiles". This is used internally by chemprop to extract features.
    """

    def __init__(self, name: str,
                 depth: int,
                 ffn_num_layers: int,
                 ffn_hidden_size: int,
                 ):
        """
        Adaptor to the chemprop model.

        Args:
            name (str): name of this model
            depth (int):  Number of message passing steps
            ffn_num_layers (int): Number of layers in FFN after MPN encoding
            ffn_hidden_size (int): Hidden dim for higher-capacity FFN (defaults to hidden_size)
        """

        super(ChempropNet, self).__init__()

        required_parameter_dict = {'data_path': "NONE", 'dataset_type': "regression"}

        parameters_dict = dict(depth=depth,
                               ffn_num_layers=ffn_num_layers,
                               ffn_hidden_size=ffn_hidden_size)

        parameters_dict.update(required_parameter_dict)

        args = TrainArgs()
        args.from_dict(parameters_dict)

        self.chemprop_model = MoleculeModel(args=args, featurizer=False)

    @staticmethod
    def _get_list_of_smiles_from_batch(batch: Batch):
        """
        This method takes a batch and extracts the list of smiles.
        It is a bit dirty, as the batch.smiles may return a list of list or a list...
        TODO: It would be better to make sure pytorch_geometric collates the smiles
           correctly as a list instead of going around its default behavior here.
        """
        smiles_array = np.array(batch.smiles).flatten()
        # numpy casts things to its own types, which breaks everything else. Bring it back to vanilla strings.
        list_smiles = [str(smiles) for smiles in smiles_array]
        return list_smiles

    def forward(self, batch: Batch):
        list_smiles = self._get_list_of_smiles_from_batch(batch)
        chemprop_output = self.chemprop_model.forward(list_smiles)
        return chemprop_output.flatten()

