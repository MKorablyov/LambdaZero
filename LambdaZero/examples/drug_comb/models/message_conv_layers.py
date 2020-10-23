from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter
import torch

########################################################################################################################
# Constants
########################################################################################################################

# Index in drug_drug_batch tuple in which the drug-drug
# edge index for the batch lies
BATCH_EDGE_IDX_IDX = 0

########################################################################################################################
# Convolution layers
########################################################################################################################

def _compute_norm(edge_idx, num_nodes=None, edge_weight=None, dtype=torch.float32):
    if num_nodes is None:
        num_nodes = torch.max(edge_idx).item()

    if edge_weight is None:
        edge_weight = torch.ones((edge_idx.shape[1],), dtype=dtype, device=edge_idx.device)

    row, col = edge_idx
    deg = degree(col, num_nodes, dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.

    return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class FourMessageConvLayer(MessagePassing):
    def __init__(self, in_drug_channels, in_prot_channels,
                 out_drug_channels, out_prot_channels, pass_d2d_msg,
                 pass_d2p_msg, pass_p2d_msg, pass_p2p_msg, data):
        """
        Graph Convolution layer with 4 types of messages (drug-drug, drug-prot, prot-drug and prot-prot)
        """
        super(FourMessageConvLayer, self).__init__(aggr='add')  # "Add" aggregation.

        # Define message passing matrices
        self.drug_to_drug_mat = torch.nn.Linear(in_drug_channels, out_drug_channels)
        self.drug_to_prot_mat = torch.nn.Linear(in_drug_channels, out_prot_channels)
        self.prot_to_drug_mat = torch.nn.Linear(in_prot_channels, out_drug_channels)
        self.prot_to_prot_mat = torch.nn.Linear(in_prot_channels, out_prot_channels)

        # Use these as a switch to turn on/off message passing between various
        # node types in the graph.
        self.pass_d2d_msg = int(pass_d2d_msg)
        self.pass_d2p_msg = int(pass_d2p_msg)
        self.pass_p2d_msg = int(pass_p2d_msg)
        self.pass_p2p_msg = int(pass_p2p_msg)

        self.use_drug_self_loop = min(1, self.pass_p2d_msg + self.pass_d2d_msg)
        self.use_prot_self_loop = min(1, self.pass_d2p_msg + self.pass_p2p_msg)

        self.self_prot_loop = torch.nn.Linear(in_prot_channels, out_prot_channels)
        self.self_drug_loop = torch.nn.Linear(in_drug_channels, out_drug_channels)

        # Compute normalization.
        row, col = torch.cat((data.ppi_edge_idx, data.dpi_edge_idx,
                              data.ddi_edge_idx, data.dpi_edge_idx[[1, 0], :]), dim=1)

        deg = degree(col, data.x_drugs.shape[0] + data.x_prots.shape[0], dtype=data.x_drugs.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Norms for each edge type
        self.ppi_norm = norm[:data.ppi_edge_idx.shape[1]]
        self.dpi_norm = norm[data.ppi_edge_idx.shape[1]: data.ppi_edge_idx.shape[1] + data.dpi_edge_idx.shape[1]]
        self.ddi_norm = norm[data.ppi_edge_idx.shape[1] + data.dpi_edge_idx.shape[1] :
                             data.ppi_edge_idx.shape[1] + data.dpi_edge_idx.shape[1] + data.ddi_edge_idx.shape[1]]
        self.pdi_norm = norm[-data.dpi_edge_idx.shape[1]:]

    def forward(self, h_drug, h_prot, data, drug_drug_batch):
        # Linearly transform node feature matrix and set to zero where relevant
        d2d_msg = self.drug_to_drug_mat(h_drug) * data.is_drug * self.pass_d2d_msg
        d2p_msg = self.drug_to_prot_mat(h_drug) * data.is_drug * self.pass_d2p_msg
        p2d_msg = self.prot_to_drug_mat(h_prot) * (1 - data.is_drug) * self.pass_p2d_msg
        p2p_msg = self.prot_to_prot_mat(h_prot) * (1 - data.is_drug) * self.pass_p2p_msg

        self_drug_loop_msg = self.self_drug_loop(h_drug) * data.is_drug * self.use_drug_self_loop
        self_prot_loop_msg = self.self_prot_loop(h_prot) * (1 - data.is_drug) * self.use_prot_self_loop

        # Propagating messages.
        drug_output = self.propagate(data.dpi_edge_idx[[1, 0], :], x=p2d_msg, norm=self.pdi_norm) + \
                      self.propagate(data.ddi_edge_idx, x=d2d_msg, norm=self.ddi_norm) + self_drug_loop_msg

        protein_output = self.propagate(data.dpi_edge_idx, x=d2p_msg, norm=self.dpi_norm) + \
                         self.propagate(data.ppi_edge_idx, x=p2p_msg, norm=self.ppi_norm) + self_prot_loop_msg

        return drug_output, protein_output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class ProtDrugMessageConvLayer(MessagePassing):
    def __init__(self, in_drug_channels, in_prot_channels, out_drug_channels, out_prot_channels, data):
        """
        Graph Convolution layer with prot-drug messages only
        """
        super(ProtDrugMessageConvLayer, self).__init__(aggr='add')  # "Add" aggregation.

        # Define message passing matrices
        self.prot_to_drug_mat = torch.nn.Linear(in_prot_channels, out_drug_channels)

        # Compute normalization.
        row, col = data.dpi_edge_idx[[1, 0], :]
        deg = degree(col, data.x_drugs.shape[0], dtype=data.x_drugs.dtype)
        deg_inv = deg.pow(-1)
        self.pdi_norm = deg_inv[col]

    def forward(self, h_drug, h_prot, data, drug_drug_batch):
        p2d_msg = self.prot_to_drug_mat(h_prot) * (1 - data.is_drug)

        # Propagating messages.
        drug_output = self.propagate(data.dpi_edge_idx[[1, 0], :], x=p2d_msg, norm=self.pdi_norm)

        return drug_output, h_prot

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class DummyMessageConvLayer(MessagePassing):
    def __init__(self, in_drug_channels, in_prot_channels, out_drug_channels, out_prot_channels, data):
        """
        Graph Convolution layer that sums prot embeddings
        """
        super(DummyMessageConvLayer, self).__init__(aggr='add')  # "Add" aggregation.

        # Compute normalization.
        row, col = data.dpi_edge_idx[[1, 0], :]
        deg = degree(col, data.x_drugs.shape[0], dtype=data.x_drugs.dtype)
        deg_inv = deg.pow(-1)
        self.pdi_norm = deg_inv[col]

    def forward(self, h_drug, h_prot, data, drug_drug_batch):
        p2d_msg = h_prot * (1 - data.is_drug)

        # Propagating messages.
        drug_output = self.propagate(data.dpi_edge_idx[[1, 0], :], x=p2d_msg, norm=self.pdi_norm)

        return drug_output, h_prot

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class ProtDrugProtProtConvLayer(MessagePassing):
    def __init__(self, in_drug_channels, in_prot_channels,
                 out_drug_channels, out_prot_channels, pass_d2d_msg,
                 pass_d2p_msg, pass_p2d_msg, pass_p2p_msg, data):
        """
        Graph Convolution layer with 4 types of messages (drug-drug, drug-prot, prot-drug and prot-prot)
        """
        super().__init__(aggr='add')  # "Add" aggregation.

        # Define message passing matrices
        self.drug_to_prot_mat = torch.nn.Linear(in_drug_channels, out_prot_channels)
        self.prot_to_prot_mat = torch.nn.Linear(in_prot_channels, out_prot_channels)
        self.self_prot_loop = torch.nn.Linear(in_prot_channels, out_prot_channels)

        # Compute normalization.
        self.num_prots = data.x_prots.shape[0]
        self.ppi_norm = _compute_norm(data.ppi_edge_idx,
                                      num_nodes=data.x_prots.shape[0],
                                      dtype=data.x_prots.dtype)

    def forward(self, h_drug, h_prot, data, drug_drug_batch):
        # Linearly transform node feature matrix and set to zero where relevant
        d2p_msg = self.drug_to_prot_mat(h_drug)
        p2p_msg = self.prot_to_prot_mat(h_prot)
        self_prot_loop_msg = self.self_prot_loop(h_prot)

        # Hack for doing something like torch.isin since pytorch hasn't actually
        # implemented isin yet.
        this_batch_drugs = drug_drug_batch[BATCH_EDGE_IDX_IDX]
        matching_drugs_dpi = (data.dpi_edge_idx[0][..., None] == this_batch_drugs.flatten()).any(-1)
        this_iter_dpi_idx = data.dpi_edge_idx[:, matching_drugs_dpi]

        # Want to send the attrs (concentrations) to modulate how strong our
        # drug -> protein messages should be
        dpi_attr = self._get_dpi_attr(this_iter_dpi_idx, this_batch_drugs, data)

        protein_output = self.propagate(this_iter_dpi_idx, x=d2p_msg, norm=dpi_attr, is_dpi=True) + \
                         self.propagate(data.ppi_edge_idx, x=p2p_msg, norm=self.ppi_norm, is_dpi=False) + \
                         self_prot_loop_msg

        return h_drug, protein_output

    # Note that a way to do the equivalent comparison to (data.ddi_edge_idx == pair).all(0)
    # for all pairs in a batch would be
    #
    #   (data.ddi_edge_idx[..., None, None] == batch_drugs).any(-1).any(-1).all(0)
    #
    def _get_dpi_attr(self, dpi_idx, batch_drugs, data):
        # Super slow but will be fine while we have batch size of just 1
        concs = torch.zeros((dpi_idx.shape[1],), device=dpi_idx.device)
        for i in range(batch_drugs.shape[0]):
            pair = batch_drugs[i]
            edge_attr = data.ddi_edge_attr[(data.ddi_edge_idx == pair.reshape(2, 1)).all(0)].squeeze()
            assert len(edge_attr.shape) == 1

            concs[(dpi_idx[0] == pair[0])] = edge_attr[0]
            concs[(dpi_idx[0] == pair[1])] = edge_attr[1]

        return concs

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs, index, dim_size, is_dpi):
        if is_dpi:
            dim_size = self.num_prots

        return scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)

