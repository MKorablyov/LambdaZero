import pytest
import torch
import numpy as np

from LambdaZero.examples.env3d.models.joint_prediction_model import BlockAngleModel

list_devices = [torch.device("cpu")]
if torch.cuda.is_available():
    list_devices.append(torch.device("cuda"))


def test_extract_attachment_node_representation(dataloader):
    torch.random.manual_seed(14123)

    hidden_dimension_size = 8

    for batch in dataloader:
        node_representations = torch.rand(batch.num_nodes, hidden_dimension_size)

        expected_attachment_node_representations = []
        cumulative_index = 0
        for batch_index, node_idx in enumerate(batch.attachment_node_idx.numpy()):
            global_index = cumulative_index + node_idx
            expected_attachment_node_representations.append(
                node_representations[global_index]
            )
            cumulative_index += np.sum(batch.batch.numpy() == batch_index)

        expected_attachment_node_representations = torch.stack(
            expected_attachment_node_representations
        )

        computed_attachment_node_representations = BlockAngleModel._extract_attachment_node_representation(
            node_representations, batch.attachment_node_idx, batch.batch
        )

        assert torch.allclose(
            computed_attachment_node_representations,
            expected_attachment_node_representations,
        )


@pytest.mark.parametrize("device", list_devices)
def test_smoke_block_angle_model(local_ray, dataloader, device):
    """
    A simple "smoke test", ie will the model even run given expected input data.
    """

    number_of_block_classes = 7
    model = BlockAngleModel(
        num_edge_features=4,
        num_hidden_features=28,
        num_edge_network_hidden_features=16,
        num_block_prediction_hidden_features=12,
        num_angle_prediction_hidden_features=12,
        number_of_block_classes=number_of_block_classes,
    )
    model.to(device)

    for batch in dataloader:
        batch = batch.to(device)
        block_logits, angle_uv = model.forward(batch)

        expected_block_logits_shape = torch.Size(
            [batch.num_graphs, number_of_block_classes]
        )
        expected_angle_uv_shape = torch.Size([batch.num_graphs, 2])

        assert block_logits.shape == expected_block_logits_shape
        assert angle_uv.shape == expected_angle_uv_shape
