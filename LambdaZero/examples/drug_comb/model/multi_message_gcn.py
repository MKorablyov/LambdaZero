import torch
from torch.nn import functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import time


class MultiMessageGraphConv(MessagePassing):
    def __init__(self, in_drug_channels, in_prot_channels, out_drug_channels, out_prot_channels, data):
        super(MultiMessageGraphConv, self).__init__(aggr='add')  # "Add" aggregation.

        # Define message passing matrices
        self.drug_to_prot_mat = torch.nn.Linear(in_drug_channels, out_prot_channels)
        self.prot_to_drug_mat = torch.nn.Linear(in_prot_channels, out_drug_channels)
        self.prot_to_prot_mat = torch.nn.Linear(in_prot_channels, out_prot_channels)

        self.self_prot_loop = torch.nn.Linear(in_prot_channels, out_prot_channels)
        self.self_drug_loop = torch.nn.Linear(in_drug_channels, out_drug_channels)

        # Compute normalization.
        row, col = torch.cat((data.ppi_edge_idx, data.dpi_edge_idx, data.dpi_edge_idx[[1, 0], :]), dim=1)
        deg = degree(col, data.x_drugs.size(0), dtype=data.x_drugs.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Norms for each edge type
        self.ppi_norm = norm[:data.ppi_edge_idx.shape[1]]
        self.dpi_norm = norm[data.ppi_edge_idx.shape[1]: data.ppi_edge_idx.shape[1] + data.dpi_edge_idx.shape[1]]
        self.pdi_norm = norm[-data.dpi_edge_idx.shape[1]:]

    def forward(self, data, h_drug, h_prot):
        # Linearly transform node feature matrix and set to zero where relevant
        d2p_msg = self.drug_to_prot_mat(h_drug) * data.is_drug[:, None]
        p2d_msg = self.prot_to_drug_mat(h_prot) * (1-data.is_drug[:, None])
        p2p_msg = self.prot_to_prot_mat(h_prot) * (1-data.is_drug[:, None])
        self_drug_loop_msg = self.self_drug_loop(h_drug) * data.is_drug[:, None]
        self_prot_loop_msg = self.self_prot_loop(h_prot) * (1-data.is_drug[:, None])

        # Propagating messages.
        drug_output = self.propagate(data.dpi_edge_idx[[1, 0], :], x=p2d_msg, norm=self.pdi_norm) + self_drug_loop_msg
        protein_output = self.propagate(data.dpi_edge_idx, x=d2p_msg, norm=self.dpi_norm) + \
                         self.propagate(data.ppi_edge_idx, x=p2p_msg, norm=self.ppi_norm) + self_prot_loop_msg

        return drug_output, protein_output

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GiantGraphMPNN(torch.nn.Module):
    def __init__(self, config, data):
        super(GiantGraphMPNN, self).__init__()

        self.device = config["device"]

        self.conv1 = MultiMessageGraphConv(data.x_drugs.shape[1], data.x_prots.shape[1], 16, 16, data)
        self.conv2 = MultiMessageGraphConv(16, 16, 16, 16, data)

        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))

        self.predictor = Parameter(1/100 * torch.randn((self.num_cell_lines, 16, 16)))

        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch):

        ##########################################
        # GNN forward pass
        ##########################################
        h_drug, h_prot = self.conv1(data, data.x_drugs, data.x_prots)
        h_drug = F.relu(h_drug)
        h_prot = F.relu(h_prot)
        h_drug, h_prot = self.conv2(data, h_drug, h_prot)

        ##########################################
        # Predict score for each pair
        ##########################################
        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1s = h_drug[drug_1s].reshape(batch_size, 1, -1)  # Trick to allow broadcasting with matmul
        h_drug_2s = h_drug[drug_2s].reshape(batch_size, -1, 1)  # Trick to allow broadcasting with matmul

        batch_score_preds = h_drug_1s.matmul(self.predictor[cell_lines]).matmul(h_drug_2s)[:, 0, 0]

        return batch_score_preds

    def loss(self, output, drug_drug_batch):

        # Using HSA scores for now
        ground_truth_scores = drug_drug_batch[2][:, 0]

        return self.criterion(output, ground_truth_scores)
