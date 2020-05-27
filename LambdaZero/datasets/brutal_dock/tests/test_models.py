import pytest
import torch
from chemprop.features import BatchMolGraph

from LambdaZero.datasets.brutal_dock.chemprop_adaptors.utils import get_chemprop_graphs_from_raw_data_dataframe
from LambdaZero.datasets.brutal_dock.models.chemprop_model import ChempropNet, OptimizedChempropNet
from LambdaZero.datasets.brutal_dock.models.message_passing_model import MessagePassingNet


@pytest.fixture
def chemprop_parameters():
    parameters = {'name': 'chemprop', 'depth': 2, 'ffn_num_layers': 2, 'ffn_hidden_size': 8}
    return parameters


@pytest.fixture
def batch_mol_graph(smiles_and_scores_dataframe):
    list_data_dict = get_chemprop_graphs_from_raw_data_dataframe(smiles_and_scores_dataframe)
    return BatchMolGraph([d['mol_graph'] for d in list_data_dict])



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


def test_chemprop_net(real_molecule_batch, chemprop_parameters):
    """
    A smoke test showing that the chemprop model runs on data of the expected shape.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch = real_molecule_batch.to(device)

    net = ChempropNet(**chemprop_parameters)
    net.to(device)

    _ = net.forward(batch)


def test_optimized_chemprop_net(batch_mol_graph, chemprop_parameters):
    """
    A smoke test showing that the chemprop model runs on data of the expected shape.
    """

    net = OptimizedChempropNet(**chemprop_parameters)

    _ = net.forward(batch_mol_graph)
