from typing import Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import NNConv, Set2Set


class MPNNBlock(nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>.
    """

    def __init__(
        self,
        num_edge_features: int = 4,
        num_hidden_features: int = 64,
        num_edge_network_hidden_features: int = 128,
        number_of_layers: int = 3,
        set2set_steps: int = 3,
        decouple_layers: bool = False
    ):
        """

        Args:
            num_edge_features (int, optional): the dimension of an edge's attribute array. Defaults to 4.
            num_hidden_features (int, optional): the dimension of node hidden representations arrays. Defaults to 64.
            num_edge_network_hidden_features (int, optional): the hidden layer dimension  of the edge network MLP.
                                                              Defaults to 128.
            number_of_layers (int, optional): number of MPNN iterations. Defaults to 3.
            set2set_steps (int, optional): number of processing steps to take in the set2set readout function.
                                           Defaults to 3.
            decouple_layers (bool, optional): if True, decouple the weights of the GCN layers. Defaults to False.

        """
        super(MPNNBlock, self).__init__()

        self.num_hidden_features = num_hidden_features
        self.number_of_layers = number_of_layers

        # minimal MLP network to use edge features in the message function, following Gilmer et al.
        edge_feature_net = nn.Sequential(
            nn.Linear(num_edge_features, num_edge_network_hidden_features),
            nn.ReLU(),
            nn.Linear(
                num_edge_network_hidden_features,
                num_hidden_features * num_hidden_features,
            ),
        )

        if decouple_layers is True:
            gcn_list = [NNConv(in_channels=num_hidden_features, 
                               out_channels=num_hidden_features,
                               nn=edge_feature_net,
                               aggr="add") for _ in range( self.number_of_layers)]
        else:
            gcn_list = [NNConv(in_channels=num_hidden_features, 
                               out_channels=num_hidden_features,
                               nn=edge_feature_net,
                               aggr="add")]

        self.aggregated_message_function = nn.ModuleList(gcn_list)

        self.gru_update_function = nn.GRU(num_hidden_features, num_hidden_features)

        self.set2set_readout = Set2Set(
            num_hidden_features, processing_steps=set2set_steps
        )

        # minimal MLP network to postprocess the output of set2set
        self.readout_net = nn.Sequential(
            nn.Linear(2 * num_hidden_features, num_hidden_features),
            nn.ReLU(),
            nn.Linear(num_hidden_features, num_hidden_features),
        )

    @staticmethod
    def _embed_node_features(node_features: torch.tensor, num_hidden_features: int):
        """
        The MPNN model of Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>  embeds
        the initial node features by zero-padding to the dimension of the hidden representation.
        This function implements this approach.

        Args:
            node_features (Tensor): node features in their original dimensions.
                                    shape = [total number of nodes in batch, number of node features]
            num_hidden_features (int): the dimension of node hidden representations arrays.

        Returns:
            hidden_embedding (Tensor): node features padded to the hidden representation dimension
                                    shape = [total number of nodes in batch, num_hidden_features]
        """

        shape = list(node_features.shape)
        assert (
            shape[-1] <= num_hidden_features
        ), f"cannot embed features in {num_hidden_features} dimensions"

        num_node_features = shape[-1]
        shape[-1] = num_hidden_features

        hidden_embedding = torch.zeros(shape)
        hidden_embedding[..., :num_node_features] = node_features

        return hidden_embedding

    def forward(self, data: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data (Batch): a batch of graphs in the pytorch-geometric format.

        Returns:
            node_representations (torch.Tensor): node level representation for all nodes
                                                 shape = [total number of nodes in batch, num_hidden_features]
            graph_representation (torch.Tensor): graph level representation array
                                                 shape = [batch size, num_hidden_features]
        """

        # embed the node features into the hidden representation space
        #   data.x is of dimensions [total number of nodes in batch, number of node features]
        #   node_embeddings  is of dimensions [total number of nodes, hidden representation dimension]
        node_embeddings = self._embed_node_features(data.x, self.num_hidden_features)

        if data.x.is_cuda:
            node_embeddings = node_embeddings.to(data.x.get_device())

        # There is a bit of tediousness below with squeezes and unsqueezes. This is caused by the GRU
        # update function that requires a "sequence length" dimension.

        # We use "unsqueeze" to create a first dimension of size 1 in "h", such that
        # h is of dimension [1, total number of nodes, hidden representation dimension]
        h = node_embeddings.unsqueeze(0)

        for i in range(self.number_of_layers):

            # dimension of aggregated_messages : [1, total number of nodes, hidden representation dimension]
            layer_idx = min(i, len(self.aggregated_message_function) - 1)
            aggregated_messages = self.aggregated_message_function[layer_idx](
                h.squeeze(0), data.edge_index, data.edge_attr
            ).unsqueeze(0)

            # From the GRU documentation:
            #       The shape of input for GRU must be (seq_len, batch, input_size)
            #       The shape of hidden for GRU must be (num_layers * num_directions, batch, hidden_size)
            # -> We iterate over layers one by one, so seq_len = 1
            # -> we consider a single layer GRU with a single direction, so num_layers * num_directions = 1
            #
            # The output of GRU, output and h, are identical when seq_len = num_layers*num_directions,
            # so we drop the output. The dimension of h is
            #           [1, total number of nodes, hidden representation dimension].
            # which is consistent with its input dimensions.
            _, h = self.gru_update_function(aggregated_messages, h)

        # We squeeze out the useless first dimension of size 1.
        # node_representations is of dimension  [total number of nodes, hidden representation dimension].
        node_representations = h.squeeze(0)

        # the readout has dimensions [batch size (ie, number of graphs in batch), 2 x hidden representation dimension]
        readout = self.set2set_readout(node_representations, data.batch)

        # the graph_representation has dimensions [batch size, hidden representation dimension]
        graph_representation = self.readout_net(readout)

        return node_representations, graph_representation
