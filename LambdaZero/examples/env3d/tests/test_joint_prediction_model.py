import torch
import numpy as np

from LambdaZero.examples.env3d.models.joint_prediction_model import BlockAngleModel


def test_extract_attachment_node_representation(dataloader):
    torch.random.manual_seed(14123)

    hidden_dimension_size = 8

    for batch in dataloader:
        node_representations = torch.rand(batch.num_nodes, hidden_dimension_size)

        expected_attachment_node_representations = []
        cumulative_index = 0
        for batch_index, node_idx in enumerate(batch.attachment_node_idx.numpy()):
            global_index = cumulative_index + node_idx
            expected_attachment_node_representations.append(node_representations[global_index])
            cumulative_index += np.sum(batch.batch.numpy() == batch_index)

        expected_attachment_node_representations = torch.stack(expected_attachment_node_representations)

        computed_attachment_node_representations = BlockAngleModel._extract_attachment_node_representation(
            node_representations, batch.attachment_node_idx, batch.batch
        )

        assert torch.allclose(computed_attachment_node_representations, expected_attachment_node_representations)
