from chemprop.args import TrainArgs
from chemprop.features import BatchMolGraph
import torch
from torch_geometric.data import Batch
import numpy as np

from chemprop.models import MoleculeModel

"""
According to chemprop/args.py, the chemprop model arguments are given by:

# Model arguments
    bias: bool = False  # Whether to add bias to linear layers
    hidden_size: int = 300  # Dimensionality of hidden layers in MPN
    depth: int = 3  # Number of message passing steps
    dropout: float = 0.0  # Dropout probability
*   activation: Literal['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'] = 'ReLU'  # Activation function
    atom_messages: bool = False  # Centers messages on atoms instead of on bonds
    undirected: bool = False  # Undirected edges (always sum the two relevant bond vectors)
    ffn_hidden_size: int = None  # Hidden dim for higher-capacity FFN (defaults to hidden_size)
    ffn_num_layers: int = 2  # Number of layers in FFN after MPN encoding
*   features_only: bool = False  # Use only the additional features in an FFN, no graph network
*   separate_val_features_path: List[str] = None  # Path to file with features for separate val set
*   separate_test_features_path: List[str] = None  # Path to file with features for separate test set
*   config_path: str = None  # Path to a .json file containing arguments. Any arguments present in the config file will override arguments specified via the command line or by the defaults.
*   ensemble_size: int = 1  # Number of models in ensemble

I indicate with "*" the variables I don't think we should consider at this time.
"""


class BaseChempropNet(torch.nn.Module):
    """
    This model class adapts the chemprop model to fit in within LambdaZero's framework.
    The forward method must be implemented based on desired data type.
    """

    @classmethod
    def _buggy_input(cls, atom_messages: bool, undirected: bool, depth: int) -> bool:
        """
        There is a bug in chemprop. The model crashes because of mismatched dimensions if
        atom_messages = True, undirected = True and depth > 1.
        """
        return atom_messages and undirected and depth > 1

    def __init__(
        self,
        name: str,
        bias: bool = False,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        atom_messages: bool = False,
        undirected: bool = False,
        ffn_hidden_size: int = None,
        ffn_num_layers: int = 2,
    ):
        """
        Adaptor to the chemprop model.

        Args:
            name (str): name of this model
            [others] : see description above
        """

        if self._buggy_input(atom_messages, undirected, depth):
            raise NotImplementedError(
                "There is a bug in Chemprop, which crashes for "
                "atom_messages = True, and undirected = True and depth > 1. Review input"
            )

        super(BaseChempropNet, self).__init__()

        required_parameter_dict = {"data_path": "NONE", "dataset_type": "regression"}

        parameters_dict = dict(
            bias=bias,
            hidden_size=hidden_size,
            depth=depth,
            dropout=dropout,
            atom_messages=atom_messages,
            undirected=undirected,
            ffn_hidden_size=ffn_hidden_size,
            ffn_num_layers=ffn_num_layers,
            num_tasks=1,
        )

        parameters_dict.update(required_parameter_dict)

        args = TrainArgs()
        args.from_dict(parameters_dict)

        self.chemprop_model = MoleculeModel(args=args, featurizer=False)

    def forward(self, *args, **kargs):
        raise NotImplementedError(
            "This class must be subclassed and a specific implementation must be created"
        )


class GeometricChempropNet(BaseChempropNet):
    """
    This chemprop adaptor assumes that the input data is a pytorch_geometric batch of molecules containing
    a field "smiles". This is used internally by chemprop to extract features.
    """

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


class MolGraphChempropNet(BaseChempropNet):
    """
    This chemprop adaptor assumes that the input data is a BatchMolGraph object (a chemprop entity).
    """

    def forward(self, batch: BatchMolGraph):
        chemprop_output = self.chemprop_model.forward(batch)
        return chemprop_output.flatten()
