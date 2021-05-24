import torch
from torch_scatter import scatter


class HMPConv(torch.nn.Module):
    """ Hierarchical Message Passing: https://arxiv.org/pdf/2006.12179.pdf"""
    def __init__(self, nodes_conv, nodes2clique_conv, clique_conv, clique2nodes_conv):
        super().__init__()
        self.nodes_conv = nodes_conv
        self.nodes2clique_conv = nodes2clique_conv
        self.clique_conv = clique_conv
        self.clique2nodes_conv = clique2nodes_conv

    def forward(self, x, edge_index, edge_attr, x_clique, node2clique_index, clique_edge_index,
                clique_edge_attr=None, node2clique_edge_attr=None, clique2node_edge_attr=None):
        # conv over nodes
        # nodes -> clique
        # conv over clique
        # clique -> nodes
        x = self.nodes_conv(x, edge_index, edge_attr)
        x_clique = x_clique + self.nodes2clique_conv(x, x_clique, node2clique_index, node2clique_edge_attr)
        x_clique = self.clique_conv(x_clique, clique_edge_index, clique_edge_attr)
        x = x + self.clique2nodes_conv(x, x_clique, node2clique_index, clique2node_edge_attr)
        return x, x_clique


class Node2CliqueConvBasic(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, x_clique, node2clique_index, node2clique_edge_attr=None):
        node_idx, clique_idx = node2clique_index
        out = scatter(x[node_idx], clique_idx, dim=0, dim_size=x_clique.size(0), reduce=self.aggr)
        out = self.lin(out)
        return out


class Clique2NodeConvBasic(torch.nn.Module):
    def __init__(self, in_channels, out_channels, aggr='mean'):
        super().__init__()
        self.aggr = aggr
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=True)

    def forward(self, x, x_clique, node2clique_index, node2clique_edge_attr=None):
        node_idx, clique_idx = node2clique_index
        out = scatter(x_clique[clique_idx], node_idx, dim=0, dim_size=x.size(0), reduce=self.aggr)
        out = self.lin(out)
        return out
