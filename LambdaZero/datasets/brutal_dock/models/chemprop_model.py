from chemprop.args import TrainArgs
from torch_geometric.data import Batch

from LambdaZero.datasets.brutal_dock.models.model_base import ModelBase
from chemprop.models import MoleculeModel


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
            ffn_num_layers (int): number of edge features. Defaults to 3.
            ffn_hidden_size (int): Hidden dim for higher-capacity FFN (defaults to hidden_size)
            ffn_num_layers (int): Number of layers in FFN after MPN encoding
        """

        super(ChempropNet, self).__init__()

        required_parameter_dict = {'data_path': "NONE", 'dataset_type': "regression"}

        parameters_dict = dict(depth=depth,
                               ffn_num_layers=ffn_num_layers,
                               ffn_hidden_size=ffn_hidden_size,
                               num_tasks=1)

        parameters_dict.update(required_parameter_dict)

        args = TrainArgs()
        args.from_dict(parameters_dict)

        self.chemprop_model = MoleculeModel(args=args, featurizer=False)

    def forward(self, batch: Batch):
        return self.chemprop_model.forward(batch.smiles)

