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
    ):
        super(BlockAngleModel, self).__init__()

        self.mpnn_block = MPNNBlock(
            num_edge_features=num_edge_features,
            num_hidden_features=num_hidden_features,
            num_edge_network_hidden_features=num_edge_network_hidden_features,
            number_of_layers=number_of_layers,
            set2set_steps=set2set_steps,
        )

    @staticmethod
    def _extract_attachment_node_representation(node_representations: Tensor,
                                                attachment_indices: Tensor,
                                                batch_indices: Tensor) -> Tensor:
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
        unique_batch_indices, counts = torch.unique(batch_indices, sorted=True, return_counts=True)

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

    def forward(self, data: Batch):

        node_representations, graph_representation = self.mpnn_block(data)
