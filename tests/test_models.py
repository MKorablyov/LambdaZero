from contextlib import contextmanager

import pytest
import torch
from chemprop.features import BatchMolGraph
import itertools

from LambdaZero.oracle_models.chemprop_model import (
    GeometricChempropNet,
    MolGraphChempropNet,
)
from LambdaZero.representation_learning.chemprop_adaptors.utils import (
    get_chemprop_graphs_from_raw_data_dataframe,
)
from LambdaZero.representation_learning.models.message_passing_model import (
    MessagePassingNet,
)

# There is a tradeoff between how many combinations we consider and how long the tests take to run.
biases = [True, False]
hidden_sizes = [5]
depths = [1, 2]
dropouts = [0.1]
atom_messages = [True, False]
undirected = [True, False]
list_ffn_hidden_sizes = [7]
list_ffn_num_layers = [3]

all_chemprop_parameters = list(
    itertools.product(
        biases,
        hidden_sizes,
        depths,
        dropouts,
        atom_messages,
        undirected,
        list_ffn_hidden_sizes,
        list_ffn_num_layers,
    )
)


@pytest.fixture
def chemprop_parameters(
    bias,
    hidden_size,
    depth,
    dropout,
    atom_messages,
    undirected,
    ffn_hidden_size,
    ffn_num_layers,
):

    parameters = {
        "name": "chemprop",
        "bias": bias,
        "hidden_size": hidden_size,
        "depth": depth,
        "dropout": dropout,
        "atom_messages": atom_messages,
        "undirected": undirected,
        "ffn_hidden_size": ffn_hidden_size,
        "ffn_num_layers": ffn_num_layers,
    }

    return parameters


@pytest.fixture(scope="session")
def batch_mol_graph(smiles_and_scores_dataframe):
    list_data_dict = get_chemprop_graphs_from_raw_data_dataframe(
        smiles_and_scores_dataframe
    )
    return BatchMolGraph([d["mol_graph"] for d in list_data_dict])


@pytest.fixture
def mpnn_model(
    number_of_node_features,
    number_of_edge_features,
    gcn_size,
    edge_hidden,
    linear_hidden,
):

    model_instantiation_parameters = dict(
        node_feat=number_of_node_features,
        edge_feat=number_of_edge_features,
        gcn_size=gcn_size,
        edge_hidden=edge_hidden,
        linear_hidden=linear_hidden,
        out_size=1,
    )

    mpnn = MessagePassingNet.create_model_for_training(model_instantiation_parameters)

    return mpnn


@pytest.mark.parametrize(
    "number_of_molecules, gcn_size, edge_hidden, linear_hidden",
    [(10, 8, 8, 8), (10, 2, 3, 4), (10, 11, 7, 5)],
)
def test_message_passing_net(mpnn_model, random_molecule_batch):
    """
    A smoke test showing that the model runs on data of the expected shape.
    """
    _ = mpnn_model.forward(random_molecule_batch)


@contextmanager
def does_not_raise():
    yield


@pytest.fixture
def input_parameter_context(chemprop_parameters):
    buggy = MolGraphChempropNet._buggy_input(
        chemprop_parameters["atom_messages"],
        chemprop_parameters["undirected"],
        chemprop_parameters["depth"],
    )

    if buggy:
        return pytest.raises(NotImplementedError)
    else:
        return does_not_raise()


@pytest.mark.parametrize(
    "bias, hidden_size, depth, dropout, atom_messages, undirected, ffn_hidden_size, ffn_num_layers",
    all_chemprop_parameters,
)
def test_chemprop_net(
    real_molecule_batch, chemprop_parameters, input_parameter_context
):
    """
    A smoke test showing that the chemprop model runs on data of the expected shape.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch = real_molecule_batch.to(device)

    with input_parameter_context:
        # chemprop is a bit nasty and half puts itself on cuda on its own. Let's make it right.
        net = GeometricChempropNet(**chemprop_parameters)
        net.to(device)

        _ = net.forward(batch)


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
