import torch

from LambdaZero.examples.env3d.models.mpnn_block import MPNNBlock


def test_mpnn_model_imbed_node_features():
    num_hidden_features = 15
    num_node_features = 7
    torch.random.manual_seed(12321)

    node_features = torch.rand(3, 5, num_node_features)
    expected_hidden_features = torch.zeros(3, 5, num_hidden_features)

    for i in range(3):
        for j in range(5):
            for k in range(num_node_features):
                expected_hidden_features[i, j, k] = node_features[i, j, k]

    computed_hidden_features = MPNNBlock._embed_node_features(
        node_features, num_hidden_features
    )

    assert torch.all(torch.eq(expected_hidden_features, computed_hidden_features))


def test_smoke_mpnn_model(local_ray, dataloader):
    """
    A simple "smoke test", ie will the model even run given expected input data.
    """

    num_hidden_features = 28
    mpnn_model = MPNNBlock(
        num_edge_features=4,
        num_hidden_features=num_hidden_features,
        num_edge_network_hidden_features=16,
    )

    for batch in dataloader:
        node_representations, graph_representation = mpnn_model.forward(batch)

        expected_graph_representation_shape = torch.Size(
            [batch.num_graphs, num_hidden_features]
        )
        expected_representations_shape = torch.Size(
            [batch.num_nodes, num_hidden_features]
        )

        assert graph_representation.shape == expected_graph_representation_shape
        assert node_representations.shape == expected_representations_shape
