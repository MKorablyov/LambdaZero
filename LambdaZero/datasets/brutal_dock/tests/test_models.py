import pytest

from LambdaZero.datasets.brutal_dock.models.chemprop_model import ChempropNet
from LambdaZero.datasets.brutal_dock.models.gin import GraphIsomorphismNet
from LambdaZero.datasets.brutal_dock.models.message_passing_model import MessagePassingNet


@pytest.fixture
def mpnn_model(number_of_node_features, number_of_edge_features, gcn_size, edge_hidden, linear_hidden):

    model_instantiation_parameters = dict(node_feat=number_of_node_features,
                                          edge_feat=number_of_edge_features,
                                          gcn_size=gcn_size,
                                          edge_hidden=edge_hidden,
                                          linear_hidden=linear_hidden,
                                          out_size=1)

    mpnn = MessagePassingNet.create_model_for_training(model_instantiation_parameters)

    return mpnn


@pytest.mark.parametrize("number_of_molecules, gcn_size, edge_hidden, linear_hidden",
                         [(10, 8, 8, 8), (10, 2, 3, 4), (10, 11, 7, 5)])
def test_message_passing_net(mpnn_model, random_molecule_batch):
    """
    A smoke test showing that the model runs on data of the expected shape.
    """
    _ = mpnn_model.forward(random_molecule_batch)


def test_chemprop_net(real_molecule_batch):
    """
    A smoke test showing that the chemprop model runs on data of the expected shape.
    """
    parameters = {'name': 'chemprop', 'depth': 2, 'ffn_num_layers': 2, 'ffn_hidden_size': 8}

    net = ChempropNet(**parameters)

    _ = net.forward(real_molecule_batch)


@pytest.fixture
def gin_model(number_of_node_features, number_of_edge_features, number_of_layers, gin_size, gin_mlp_hidden_size,
              gin_hidden_size, linear_hidden):

    model_instantiation_parameters = dict(node_feat=number_of_node_features,
                                          edge_feat=number_of_edge_features,
                                          gin_layers=number_of_layers,
                                          gin_size=gin_size,
                                          gin_mlp_hidden=gin_mlp_hidden_size,
                                          gin_hidden=gin_hidden_size,
                                          linear_hidden=linear_hidden,
                                          out_size=1)

    gin = GraphIsomorphismNet.create_model_for_training(model_instantiation_parameters)

    return gin


@pytest.mark.parametrize("number_of_molecules, number_of_layers, gin_size, gin_mlp_hidden_size, gin_hidden_size," +
                         "linear_hidden",
                         [(10, 1, 4, 8, 8, 8), (10, 2, 2, 2, 2, 4), (10, 2, 11, 7, 7, 5)])
def test_graph_isomorphism_net(gin_model, random_molecule_batch):
    """
    A smoke test showing that the model runs on data of the expected shape.
    """
    _ = gin_model.forward(random_molecule_batch)
