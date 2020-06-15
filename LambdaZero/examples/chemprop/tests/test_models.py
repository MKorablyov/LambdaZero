import pytest
import torch

from LambdaZero.oracle_models.chemprop_model import MolGraphChempropNet
from tests.test_models import all_chemprop_parameters


@pytest.mark.parametrize(
    "bias, hidden_size, depth, dropout, atom_messages, undirected, ffn_hidden_size, ffn_num_layers",
    all_chemprop_parameters,
)
def test_optimized_chemprop_net(
    batch_mol_graph, chemprop_parameters, input_parameter_context
):
    """
    A smoke test showing that the chemprop model runs on data of the expected shape.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with input_parameter_context:
        # chemprop is a bit nasty and half puts itself on cuda on its own. Let's make it right.
        net = MolGraphChempropNet(**chemprop_parameters)
        net.to(device)

        _ = net.forward(batch_mol_graph)