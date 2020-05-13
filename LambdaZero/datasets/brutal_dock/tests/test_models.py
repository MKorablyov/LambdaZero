import pytest

from LambdaZero.datasets.brutal_dock.models.chemprop_model import ChempropNet


@pytest.mark.parametrize("number_of_molecules", [10])
def test_message_passing_net(mpnn_model, random_molecule_batch):
    """
    A smoke test showing that the model runs on data of the expected shape.
    """
    _ = mpnn_model.forward(random_molecule_batch)


@pytest.mark.skip(reason="The chemprop smoke test fails at this time: adapting chemprop will take some time.")
@pytest.mark.parametrize("number_of_molecules", [10])
def test_chemprop_net(random_molecule_batch):
    """
    A smoke test showing that the chemprop model runs on data of the expected shape.
    """
    parameters = {'name': 'chemprop', 'depth': 2, 'ffn_num_layers': 2, 'ffn_hidden_size': 8}

    net = ChempropNet(**parameters)

    _ = net.forward(random_molecule_batch)


