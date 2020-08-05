from typing import Tuple

import torch
from torch import nn, Tensor
from torch_geometric.data import Batch

from LambdaZero.examples.env3d.models.mpnn_block import MPNNBlock


class BlockAngleModel(nn.Module):
    def __init__(
        self,
        num_edge_features: int = 4,
        num_hidden_features: int = 64,
        num_edge_network_hidden_features: int = 128,
        number_of_layers: int = 3,
        set2set_steps: int = 3,
        num_block_prediction_hidden_features: int = 128,
        num_angle_prediction_hidden_features: int = 128,
        number_of_block_classes: int = 105,
    ):
        """



        Args:
            num_edge_features (int, optional): the dimension of an edge's attribute array for MPNN block. Defaults to 4.
            num_hidden_features (int, optional): the dimension of node hidden representations arrays for MPNN block.
                                                 Defaults to 64.
            num_edge_network_hidden_features (int, optional): the hidden layer dimension  of the edge network MLP
                                                              for MPNN block. Defaults to 128.
            number_of_layers (int, optional): number of MPNN iterations. Defaults to 3.
            set2set_steps (int, optional): number of processing steps to take in the set2set readout function for
                                            MPNN block. Defaults to 3.

            num_block_prediction_hidden_features (int, optional): dimension of block prediction hidden layer in single
                                                                  layer MLP prediction head. Defaults to 128.
            num_angle_prediction_hidden_features (int, optional): dimension of angle prediction hidden layer in single
                                                                  layer MLP prediction head. Defaults to 128.
            number_of_block_classes (int, optional): number of target classes. Default to 105. This default
                                                     derives from the assumption that we are using the 105 blocks
                                                     vocabulary.
        """
        super(BlockAngleModel, self).__init__()

        self.mpnn_block = MPNNBlock(
            num_edge_features=num_edge_features,
            num_hidden_features=num_hidden_features,
            num_edge_network_hidden_features=num_edge_network_hidden_features,
            number_of_layers=number_of_layers,
            set2set_steps=set2set_steps,
        )

        # we concatenate the attachment node representation (dim num_hidden_features),
        # the graph level representation (dim num_hidden_features) and n_axis (dim 3)
        concatenated_representation_dimension = 2 * num_hidden_features + 3

        self.block_prediction_head = nn.Sequential(
            nn.Linear(
                concatenated_representation_dimension,
                num_block_prediction_hidden_features,
            ),
            nn.ReLU(),
            nn.Linear(num_block_prediction_hidden_features, number_of_block_classes),
        )

        # the angle prediction will return (u, v), which can be normalized to give cos(\theta) and sin(\theta)
        angle_prediction_output_dimension = 2

        self.angle_prediction_head = nn.Sequential(
            nn.Linear(
                concatenated_representation_dimension,
                num_angle_prediction_hidden_features,
            ),
            nn.ReLU(),
            nn.Linear(
                num_angle_prediction_hidden_features, angle_prediction_output_dimension
            ),
        )

    @staticmethod
    def _extract_attachment_node_representation(
        node_representations: Tensor, attachment_indices: Tensor, batch_indices: Tensor
    ) -> Tensor:
        """
        Extract the node representation for the attachment node, as described by
        its index.

        Args:
            node_representations (Tensor): node hidden representations
                                           shape = [total number of nodes in batch, num_hidden_features]
             attachment_indices (Tensor): node hidden representations
                                           shape = [batch size]
            batch_indices (Tensor): array on integers indicating to which graph in a batch in a batch
                                    each node belongs.
                                    shape = [total number of nodes in batch]
        Returns:
            attachment_node_representation (Tensor): attachment node representation
                                           shape = [batch size, num_hidden_features]

        """

        # batch_indices is an array containing (number nodes in graph) repetitions of the batch index,
        # repeated for all the graphs in the batch. For example:
        #  batch_indices = [0, 0, 0, 1, 1]
        #  would be a batch of 2 graphs, the first one having 3 nodes and the second one 2 nodes.
        # We extract the unique batch indices and their counts, which indicate how many nodes are
        # present in each graph belonging to the graph.
        unique_batch_indices, counts = torch.unique(
            batch_indices, sorted=True, return_counts=True
        )

        # the node_representations tensor is a stacking of all node arrays for graphs belonging to the batch.
        # For example, again for batch_indices = [0, 0, 0, 1, 1],
        #      node_representations  = [   ---  v0 of graph 0 ----]
        #                              [   ---  v1 of graph 0 ----]
        #                              [   ---  v2 of graph 0 ----]
        #                              [   ---  v0 of graph 1 ----]
        #                              [   ---  v1 of graph 1 ----]
        #
        # In order to get vector m of graph n, we must extract the global index of the vector in the stack,
        # which is given by
        #               absolute index = (sum of all nodes up to graph n-1) + relative index
        #
        # For example, to get v1  of graph1, the absolute index is
        #               absolute index = (sum of all nodes up to graph 0 := 3) + (relative index := 1) = 4.

        cumulative_counts = torch.cumsum(counts, dim=0)

        zero = torch.tensor([0], requires_grad=False)
        sum_of_all_nodes_up_to_graph_nm1 = torch.cat([zero, cumulative_counts[:-1]])

        global_indices = sum_of_all_nodes_up_to_graph_nm1 + attachment_indices
        attachment_node_representation = node_representations[global_indices]

        return attachment_node_representation

    def forward(self, data: Batch) -> Tuple[Tensor, Tensor]:
        """

        Args:
            data (Batch): pytorch-geometric batch of graphs, assumed to contain the needed extra information
                          for this task.
        Returns:
            blocks_logits (Tensor): the unormalised probabilities for every block.
                                    shape = [batch size, number_of_block_classes]
            angle_uv (Tensor): the unormalized u, v values that can be used to predict cos(theta) and sin(theta)
                                    shape = [batch size, 2]
        """
        node_representations, graph_representation = self.mpnn_block(data)

        attachment_node_representations = self._extract_attachment_node_representation(
            node_representations, data.attachment_node_idx, data.batch
        )

        # change type to float32 to harmonize with the other arrays
        n_axis = data.n_axis.type(torch.Tensor)

        concatenated_representation = torch.cat(
            [attachment_node_representations, graph_representation, n_axis], axis=1
        )

        blocks_logits = self.block_prediction_head.forward(concatenated_representation)

        angle_uv = self.angle_prediction_head.forward(concatenated_representation)

        return blocks_logits, angle_uv
