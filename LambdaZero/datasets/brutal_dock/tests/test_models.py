from LambdaZero.datasets.brutal_dock.models import MessagePassingNet


def test_message_passing_net(number_of_node_features, random_molecule_batch):
    """
    A smoke test showing that the model runs on data of the expected shape.
    """

    mpnn = MessagePassingNet(num_feat=number_of_node_features)

    _ = mpnn.forward(random_molecule_batch)


