import torch
from torch.nn import functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import time


class MultiMessageGraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MultiMessageGraphConv, self).__init__(aggr='add')  # "Add" aggregation.

        # Define message passing matrices
        self.drug_to_prot_mat = torch.nn.Linear(in_channels, out_channels)
        self.prot_to_drug_mat = torch.nn.Linear(in_channels, out_channels)
        self.prot_to_prot_mat = torch.nn.Linear(in_channels, out_channels)
        self.self_loop = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, is_drug):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Linearly transform node feature matrix.
        d2p_msg = self.drug_to_prot_mat(x)
        p2d_msg = self.prot_to_drug_mat(x)
        p2p_msg = self.prot_to_prot_mat(x)
        self_loop_msg = self.self_loop(x)

        # Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Compute types of edges
        tail_is_drug = is_drug[row]
        head_is_drug = is_drug[col]
        # prot-prot=0, drug-prot=1, prot-drug=2
        edge_type = tail_is_drug + 2 * head_is_drug

        # Get indices of edges corresponding to each type
        p2p_edge_index = edge_index[:, edge_type == 0]
        d2p_edge_index = edge_index[:, edge_type == 1]
        p2d_edge_index = edge_index[:, edge_type == 2]

        # Propagating messages.
        return self.propagate(p2p_edge_index, x=p2p_msg, norm=norm[edge_type == 0]) + \
               self.propagate(d2p_edge_index, x=d2p_msg, norm=norm[edge_type == 1]) + \
               self.propagate(p2d_edge_index, x=p2d_msg, norm=norm[edge_type == 2]) + \
               self_loop_msg

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GiantGraphMPNN(torch.nn.Module):
    def __init__(self, config):
        super(GiantGraphMPNN, self).__init__()

        self.device = config["device"]

        self.conv1 = MultiMessageGraphConv(config["in_channels"], 16)
        self.conv2 = MultiMessageGraphConv(16, 16)

        self.num_cell_lines = config["num_cell_lines"]

        self.predictor = Parameter(1/100 * torch.randn((self.num_cell_lines, 16, 16)))

        self.criterion = torch.nn.MSELoss()

    def forward(self, data, conv_edges, drug_drug_batch):

        is_drug = data.x[:, 0]  # The first column of x indicates nodes that are drugs

        ##########################################
        # GNN forward pass
        ##########################################
        x = self.conv1(data.x, conv_edges, is_drug)
        x = F.relu(x)
        x = self.conv2(x, conv_edges, is_drug)

        ##########################################
        # Predict score for each pair
        ##########################################
        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        x_drug_1s = x[drug_1s].reshape(batch_size, 1, -1)  # Trick to allow broadcasting with matmul
        x_drug_2s = x[drug_2s].reshape(batch_size, -1, 1)  # Trick to allow broadcasting with matmul

        batch_score_preds = x_drug_1s.matmul(self.predictor[cell_lines]).matmul(x_drug_2s)[:, 0, 0]

        return batch_score_preds

    def loss(self, output, drug_drug_batch):

        # Using HSA scores for now
        ground_truth_scores = drug_drug_batch[2][:, 3]

        return self.criterion(output, ground_truth_scores)
