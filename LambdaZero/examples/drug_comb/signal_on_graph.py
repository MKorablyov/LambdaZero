from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch


# outline
# 1. Sample one drug pair at a time
# 2. Anonymous node ids to condition messages
# 3. Concatenate the whole graph


# class PPIgraph(torch.nn.module)
#   PPI

# class CellSimulator_v1(ppi_edges, ppi_feat,  dpi_edges, dpi_feat)
#   self.ppi_edges = ...
#   self.drugs = ....

#   def message_ppi()
#       #
#   def message_dpi()
#       #
#   def aggr_protein()
#        #

#   def embed_cell_state():
#       # concatenate PPI # 20,000 * 32 -> 0.5M
#       # mlp
#       # return embedding

#   def compute_drugs(idxs, concentrations)
#       # message_dpi
#       # message_ppi
#       # aggr_protein
#       # embed_cell_state
#       # return inhibition

# todo: cell line is a condition on every message
#


# class DummyMessageConvLayer(MessagePassing):
#     def __init__(self, in_drug_channels, in_prot_channels, out_drug_channels, out_prot_channels, data):
#         """
#         Graph Convolution layer that sums prot embeddings
#         """
#         super(DummyMessageConvLayer, self).__init__(aggr='add')  # "Add" aggregation.
#         # Compute normalization.
#         row, col = data.dpi_edge_idx[[1, 0], :]
#         deg = degree(col, data.x_drugs.shape[0], dtype=data.x_drugs.dtype)
#         deg_inv = deg.pow(-1)
#         self.pdi_norm = deg_inv[col]
#
#     def forward(self, h_drug, h_prot, data):
#         #p2d_msg = h_prot * (1 - data.is_drug)
#         # Propagating messages.
#         #drug_output = self.propagate(data.dpi_edge_idx[[1, 0], :], x=p2d_msg, norm=self.pdi_norm)
#         #return drug_output, h_prot
#         pass
#
#
#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j