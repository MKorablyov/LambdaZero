from collections import defaultdict
from torch_geometric.nn import GCNConv, GATConv
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
import torch
import random

class SubgraphEmbeddingRegressorModel(torch.nn.Module):
    """A model which serves to predict synergy scores of drug combinations.

    The model to do synergy score prediction for drug pairs.  For easier
    experimentation, this class contains a number of switches controlling how to
    do prediction.  In general the architecture looks like the following:

        1) Pass all subgraphs through two message passing layers to create
           embeddings for the nodes in all graphs.  The second message passing
           layer has residual connections from the first layer. Here, features
           for all subgraphs are shared for memory efficiency.
        2) Embeddings for nodes are averaged according to subgraph
           assignments, thereby creating graph embeddings for each
           subgraph which are simply the average of all the graph's
           constituent nodes.
        3) Given graph embeddings for each drug's subgraph, we do final
           prediction in one of two ways.
            A) We use the dot product, such that if we have two drugs
               :math:`d, d'` and their embeddings :math:`x_d, x_{d'}`, then
               we compute the prediction as
               :math:`\hat{y}_{d, d'} = \langle x_d, x_{d'}^T \rangle`.
            B) We use an MLP for final prediction.  Here, prediction is done
               by concatenating drug embeddings :math:`x_d, x_{d'}` together
               and feeding them first to a shared linear layer.  Then, given
               a cell line :math:`c` for a given drug pair :math:`d, d'`, we
               pass this drug pair's result from the first layer through a
               final linear layer specific to cell line :math:`c`.  The results
               of the final layer become the final predictions.

    Attributes
    ----------
    out_channels : int
        The number of values to predict for each drug pairr.
    conv1 : torch.nn.conv.MessagePassing
        The first layer in the GNN.
    conv2 : torch.nn.conv.MessagePassing
        The second layer in the GNN.
    conv_dropout : torch.nn.Dropout
        A dropout module placed in front of conv2.
    linear_dropout : torch.nn.Dropout
        A dropout layer placed in front of the final linear layer
        (if using mlp for final prediction).
    skip_gcn : bool
        If True then we skip the GNN and place protein features directly
        into final prediction.
    shared_lin_lyr : torch.nn.Linear
        Shared linear layer for prediction using MLP.
    cell_line_to_regressor : Dict[int, torch.nn.Linear]
        A dictionary mapping cell lines to their final, separate, linear layer.
    _final_predictor : Function[np.ndarray]
        The function to be used for final prediiction.  This can either be
        the method _pred_mlp or _pred_dot_product, such that we either
        do final prediction with an mlp or the dot product.

    """
    def __init__(self, config, cell_lines):
        """The init method for the model.

        Note that the config parameter holds most all of the actual arguments,
        so we simply include those in the Arguments documentation.

        Arguments
        ---------
        cell_lines : List[int]
            A list containing the cell lines that exist in the data set.  These
            cell lines are simply integer encodings of the cell lines as they
            are only to be used as keys to control branching throughout code paths.
        config : Dict
            This dictionary contains all the arguments actually used for the
            method, so we describe all keys in the dictionary here.
        out_channels : int
            The number of values to predict for each drug pairr.
        use_gat : bool
            If True, then use graph attention layers for the MPNN.  Note that
            right now the batching procedure uses way too much memory for the
            GAT layers to not OOM.  It is prescient to set use_gat to False
            here and use GCN layers so that the model does not OOM.
        in_channels : int
            The length of the input feature vectors.
        embed_channels : int
            The desired length of the embedding vectors.
        num_heads : int
            The number of heads to use for the GAT layers.
        conv_dropout_rate : float (in inclusive interval [0.0, 1.0])
            Dropout rate for the dropout placed in front of the second
            message passing layer.
        linear_dropout_rate : float (in inclusive interval [0.0, 1.0])
            Dropout rate for the dropout placed in front of the separate
            linear layers.
        skip_gcn : bool
            If True, skip the message passing layers and put the input
            features directly into final prediction. Otherwise, use the
            message passing layers.
        prediction_type : {'mlp', 'dot_product'}
            Controls whether to do final prediction using an mlp or
            the simple dot product.
        regressor_hidden_channels : int
            The number of channels that the shared linear layer will compute.
        out_channels : int
            The number of channels for final prediction.

        Raises
        ------
        KeyError
            Raised if prediction_type is not one of 'mlp' or 'dot_product'.
        """
        super().__init__()

        self.out_channels = config["out_channels"]
        if config['use_gat']:
            self.conv1 = GATConv(config["in_channels"], config["embed_channels"],
                                 heads=config["num_heads"], concat=False)
            self.conv2 = GATConv(config["embed_channels"], config["embed_channels"],
                                 heads=config["num_heads"], concat=False)
        else:
            self.conv1 = GCNConv(config["in_channels"], config["embed_channels"])
            self.conv2 = GCNConv(config["embed_channels"], config["embed_channels"])

        self.conv_dropout = torch.nn.Dropout(config['conv_dropout_rate'])
        self.linear_dropout = torch.nn.Dropout(config['linear_dropout_rate'])

        self.skip_gcn = config['skip_gcn']

        if config['prediction_type'] == 'mlp':
            self.shared_lin_lyr = torch.nn.Linear(2 * config["embed_channels"],
                                                  config["regressor_hidden_channels"])

            # The input to the regressor will be the concatenation of two graph
            # embeddings, so take the in channels here to be 2 times the embedding size
            self.cell_line_to_regressor = {
                cell_line.item(): torch.nn.Linear(config["regressor_hidden_channels"],
                                                  config["out_channels"])
                for cell_line in cell_lines
            }

        final_predictors = {
            'mlp': self._pred_mlp,
            'dot_product': self._pred_dot_product,
        }

        self._final_predictor = final_predictors[config['prediction_type']]

    def to(self, device):
        new_model = super().to(device)

        if hasattr(new_model, 'cell_line_to_regressor'):
            for cell_line, regressor in new_model.cell_line_to_regressor.items():
                new_model.cell_line_to_regressor[cell_line] = regressor.to(device)

        return new_model

    def forward(self, x, drug_drug_batch, edge_attr, edge_cell_lines, sg_edge_index, sg_nodes, sg_avging_idx):
        """Does the forward pass for the model.

        Arguments
        ---------
        x : torch.tensor
            The input feature matrix for the proteins.  The tensor is of
            shape (num_proteins, num_protein_features).
        drug_drug_batch : torch.tensor
            An edge index for the drug-drug batch.  This tensor of shape
            (2, num_batch_drug_drug_edges) then represents the drug-drug edges
            within the batch.
        edge_cell_lines : torch.LongTensor
            A tensor of shape (num_batch_drug_drug_edges,) holding the integer
            encoded cell line index for the various edges in the batch.  More
            precisely, the ith item of this tensor is the cell line for the ith
            edge in drug_drug_batch.
        sg_edge_index : torch.LongTensor
            The edge index for all the edges in the subgraphs of the batch's drugs.
            Write the set of edges in drug_drug_batch as :math:`B`.  Then write
            the set of drugs in :math:`B` as :math:`D_B = {d | (d, d') or (d', d) \in B}`.
            Following the notation of the docs for the method _build_subgraph in
            transforms.py, sg_edge_index is the union of the edges of the graphs in
            :math:`M_B`, that is, :math:`sg_edge_index = \bigcup_{d \in D_B} E_d`.
            Thus, sg_edge_index is a tensor of shape (2, :math:`|\bigcup_{d \in D_B} E_d|`).
        sg_nodes : torch.LongTensor
            The indices of the protein nodes in each of the batch's drugs subgraphs.
            Note that these indices are specified relative to the global ordering
            of protein nodes as returned by the construction of the DrugCombDb object.
            We follow the notation adopted in the explanation of sg_edge_index here.
            sg_nodes can be represented by the following set allowing duplicates as
            :math:`sg_nodes = \bigcup_{d \in D_B} V_d`.  Thus, sg_nodes is of shape
            (:math:`|\bigcup_{d \in D_B} V_d|`,).  Note that the ordering of the protein
            indices in this tensor are in order and relative to the drug's index. E.g.,
            if :math:`D_B = {1, 4, 7}` and :math:`V_1 = {10, 11, 17}, V_4 = {8, 11},
            V_7 = {9, 10, 14}`, then sg_nodes will be the tensor:
                torch.tensor([10, 11, 17, 8, 11, 9, 10, 14])

            That is, sg_nodes will be the concatenation of :math:`V_1, V_2, V_3`.
        sg_avging_idx : torch.LongTensor
            The indices in :math:`D_B` that the items of sg_nodes belong to.
            The tensor is then the same shape as sg_nodes,
            (:math:`|\bigcup_{d \in D_B} V_d|`,).  In the sg_nodes example, drug 1
            is at index 0 of :math:`D_B`, drug 4 is at index 1, and drug 7 is at
            index 2.  Note that the first three nodes in the sg_nodes example
            above are for drug 1, the next two are for drug 4, and the final
            three are for drug 7.  In this case, sg_avging_idx would be the tensor:
                torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

            sg_avging_idx is used by the method scatter_mean to do efficient
            protein node embedding averaging when obtaining graph embeddings.
        Returns
        -------
        torch.tensor
            A tensor of shape (num_drug_pairs, out_channels) containing
            synergy score predictions for each drug pair.
        """
        prot_embeds = x
        if not self.skip_gcn:
            h1 = F.relu(self.conv1(x, sg_edge_index))
            h2 = F.relu(self.conv2(self.conv_dropout(h1), sg_edge_index)) + h1
            prot_embeds = h2

        node_embeds = prot_embeds[sg_nodes]
        graph_embeds = scatter_mean(node_embeds, sg_avging_idx, dim=0)

        return self._pred_with_graph_embeds(graph_embeds, drug_drug_batch, edge_attr, edge_cell_lines)

    def _pred_with_graph_embeds(self, graph_embeds, drug_drug_batch, edge_cell_lines):
        """Extract drug pair embeddings and do final prediction.

        Note: we follow the mathematical notation of earlier docstrings here.

        Parameters
        ----------
        graph_embeds : torch.FloatTensor
            A tensor of shape (:math:`|D_B|`, embedding_size) representing the
            graph embedding for each drug :math:`d \in D_B`.  Observe that this
            tensor is indexed as 0 to :math:`|D_B| - 1`.
        drug_drug_batch : torch.tensor
            An edge index for the drug-drug batch.  This tensor of shape
            (2, num_batch_drug_drug_edges) then represents the drug-drug edges
            within the batch.  Note that the indices of the drug nodes in the
            edges here are with respect to the global drug ordering.
        edge_cell_lines : torch.LongTensor
            A tensor of shape (num_batch_drug_drug_edges,) holding the integer
            encoded cell line index for the various edges in the batch.  More
            precisely, the ith item of this tensor is the cell line for the ith
            edge in drug_drug_batch.

        Returns
        -------
        torch.tensor
            A tensor of shape (num_drug_pairs, out_channels) containing
            synergy score predictions for each drug pair.

        Notes
        -----
        The arguments graph_embeds and drug_drug_batch are *not* indexed the
        same.  All elements of drug_drug_batch are drugs :math:`d` in the
        sorted set :math:`D_B`.  On the other hand, the shape of graph_embeds
        is only (:math:`|D_B|`, embedding_size).  Thus, while :math:`D_B` could
        be, e.g., :math:`D_B = {1, 4, 7}`, graph_embeds would only contain three
        rows.  As the ith row of graph_embeds corresponds to the ith smallest
        integer drug in :math:`D_B`, the first row would correspond to drug 1,
        the second row to drug 4, and the third to drug 7.  However, due to this
        indexing discrepancy, we must do extra processing to ensure that the
        indices in drug_drug_batch match the indexing of graph_embeds.  This
        computation is represented by the code using np.unique and np.digitize.
        """
        # Quantize drug drug batch so indices match graph_embeds
        drug_bins = np.unique(drug_drug_batch.cpu().flatten()) + 1
        drug_drug_batch = torch.from_numpy(np.digitize(drug_drug_batch.cpu(), drug_bins))

        from_drug_embeds = graph_embeds[drug_drug_batch[0, :]]
        to_drug_embeds = graph_embeds[drug_drug_batch[1, :]]

        return self._final_predictor(from_drug_embeds, to_drug_embeds, edge_attr, edge_cell_lines)

    def _pred_mlp(self, from_drug_embeds, to_drug_embeds, edge_attr, edge_cell_lines):
        """
        Concatenate the embeddings then feed them through a shared linear layer,
        then finally through a separate linear layer per cell line.
        """
        x = None
        if bool(random.getrandbits(1)):
            x = torch.cat((from_drug_embeds, to_drug_embeds), dim=1)
        else:
            x = torch.cat((to_drug_embeds, from_drug_embeds), dim=1)

        if edge_attr.shape[1] != 0:
            x = torch.cat((x, edge_attr), dim=1)

        x = self.shared_lin_lyr(x)
        x = self.linear_dropout(x)

        cell_line_to_idx = defaultdict(list)
        for i, cell_line in enumerate(edge_cell_lines):
            cell_line_to_idx[cell_line.item()].append(i)

        preds = torch.empty((from_drug_embeds.shape[0], self.out_channels),
                            device=from_drug_embeds.device)
        for cell_line, cell_line_idxs in cell_line_to_idx.items():
            regressor = self.cell_line_to_regressor[cell_line]
            preds[cell_line_idxs] = regressor(x[cell_line_idxs])

        return preds

    def _pred_dot_product(self, from_drug_embeds, to_drug_embeds, edge_attr, edge_cell_lines):
        return torch.sum(from_drug_embeds * to_drug_embeds, dim=1)

