import torch
import torch.nn as nn

from torch_geometric.nn import Set2Set

from torch_geometric.nn import NNConv
from LambdaZero.models.pna import PNAConv
from LambdaZero.models.hmp import HMPConv, Node2CliqueConvBasic, Clique2NodeConvBasic

from functools import partial


class NodeEmb(torch.nn.Module):
    def __init__(self, max_z, node_emb_size, node_hidden_channels):
        super().__init__()
        self.node_emb_layer = nn.Embedding(num_embeddings=max_z, embedding_dim=node_emb_size, padding_idx=0)
        self.node_lin = nn.Linear(node_emb_size, node_hidden_channels)

    def forward(self, graph):
        node_features = torch.relu(self.node_emb_layer(graph.z))
        node_features = torch.relu(self.node_lin(node_features))
        return node_features


class CliqueEmb(NodeEmb):
    def __init__(self, max_vocab_value, clique_emb_size, clique_hidden_channels):
        super().__init__(max_vocab_value, clique_emb_size, clique_hidden_channels)

    def forward(self, graph):
        clique_features = torch.relu(self.node_emb_layer(graph.x_clique))
        clique_features = torch.relu(self.node_lin(clique_features))
        return clique_features


class EdgeEmb(torch.nn.Module):
    def __init__(self, emb_type, max_edge_value, edge_emb_size):
        assert emb_type in ['blank', 'bond_type', 'distance', 'bond_type+distance'], "type should be either of 'blank', 'bond_type', 'distance', 'bond_type+distance'"
        super().__init__()
        self.emb_type = emb_type
        if emb_type == 'blank':
            self.blank_emb = nn.Embedding(num_embeddings=1, embedding_dim=edge_emb_size, padding_idx=0)
        elif emb_type == 'bond_type':
            self.bond_type_emb = nn.Embedding(num_embeddings=max_edge_value, embedding_dim=edge_emb_size, padding_idx=0)
        elif emb_type == 'distance':
            self.distance_emb = nn.Linear(1, edge_emb_size)
        else:  # self.emb_type == 'bond_type+distance'
            self.bond_type_emb = nn.Embedding(num_embeddings=max_edge_value, embedding_dim=edge_emb_size, padding_idx=0)
            self.distance_emb = nn.Linear(1, edge_emb_size)
            self.lin_reduce = nn.Linear(2*edge_emb_size, edge_emb_size)

    def forward(self, graph):
        if self.emb_type == 'blank':
            edge_attr = graph.edge_index.new_zeros((graph.edge_index.size(1), 1))
            edge_features = self.blank_emb(edge_attr)
        elif self.emb_type == 'bond_type':
            edge_features = self.bond_type_emb(graph.bond_type)
        elif self.emb_type == 'distance':
            edge_features = self.distance_emb(graph.abs_distances.unsqueeze(1))
        else:  # self.emb_type == 'bond_type+distance'
            bond_type_features = torch.relu(self.bond_type_emb(graph.bond_type))
            distance_features = torch.relu(self.distance_emb(graph.abs_distances.unsqueeze(1)))
            edge_features = self.lin_reduce(torch.cat((bond_type_features, distance_features), dim=-1))
        return edge_features


# TODO: figure something beyond blank embedding
class CliqueEdgeEmb(torch.nn.Module):
    def __init__(self, clique_edge_emb_size):
        super().__init__()
        self.blank_emb = nn.Embedding(num_embeddings=1, embedding_dim=clique_edge_emb_size, padding_idx=0)

    def forward(self, graph):
        clique_edge_attr = graph.clique_edge_index.new_zeros((graph.clique_edge_index.size(1), 1))
        return torch.relu(self.blank_emb(clique_edge_attr))


class GlobalPooling(torch.nn.Module):
    def __init__(self, hierarchical, node_channels, clique_channels=None):
        super().__init__()
        self.hierarchical = hierarchical
        if hierarchical:
            self.nodes_pooling = Set2Set(node_channels, processing_steps=3)
            self.clique_pooling = Set2Set(clique_channels, processing_steps=3)
            self.lin_reduce = nn.Sequential(
                nn.Linear(2 * (node_channels + clique_channels), (node_channels + clique_channels)),
                nn.ReLU(),
                nn.Linear((node_channels + clique_channels), 1))
        else:
            self.nodes_pooling = Set2Set(node_channels, processing_steps=3)
            self.lin_reduce = nn.Sequential(
                nn.Linear(2 * node_channels, node_channels),
                nn.ReLU(),
                nn.Linear(node_channels, 1))

    def forward(self, x, batch_index, x_clique=None, clique_batch_index=None):
        if self.hierarchical:
            assert x_clique is not None and clique_batch_index is not None, "you should supply features from clique for hierarchical pooling"
            out_nodes = self.nodes_pooling(x, batch_index)
            out_clique = self.clique_pooling(x_clique, clique_batch_index)
            out = self.lin_reduce(torch.cat((out_nodes, out_clique), dim=-1))
        else:
            out = self.lin_reduce(self.nodes_pooling(x, batch_index))
        return out


# TODO: pretty sure it can be more general and compact
class NNConvPlainModel(torch.nn.Module):
    def __init__(self, n_inter_layers, node_hidden_channels, node_output_channels, edge_emb_size,
                 flow='source_to_target'):
        super().__init__()
        self.model = nn.ModuleList()
        for _ in range(n_inter_layers-1):
            self.model.append(NNConv(in_channels=node_hidden_channels,
                                     out_channels=node_hidden_channels,
                                     aggr='mean',
                                     flow=flow,
                                     nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                        nn.ReLU(),
                                                        nn.Linear(node_hidden_channels, node_hidden_channels * node_hidden_channels)])))

        self.model.append(NNConv(in_channels=node_hidden_channels,
                                 out_channels=node_output_channels,
                                 aggr='mean',
                                 flow=flow,
                                 nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                    nn.ReLU(),
                                                    nn.Linear(node_hidden_channels, node_hidden_channels * node_output_channels)])))

    def forward(self, node_features, edge_index, edge_features):
        for layer in self.model:
            node_features = layer(node_features, edge_index, edge_features)
            node_features = torch.relu(node_features)
        return node_features


class NNConvHierarchicalModel(torch.nn.Module):
    def __init__(self, n_inter_layers,
                 node_hidden_channels, node_output_channels, edge_emb_size,
                 clique_hidden_channels, clique_output_channels, clique_edge_emb_size,
                 flow='source_to_target'):
        super().__init__()
        self.model = nn.ModuleList()

        # distribute complexity of reading comprehension
        nodes_conv = partial(NNConv, in_channels=node_hidden_channels, out_channels=node_hidden_channels, aggr='mean', flow=flow)
        nodes2clique_conv = partial(Node2CliqueConvBasic, in_channels=node_hidden_channels, out_channels=clique_hidden_channels)
        clique_conv = partial(NNConv, in_channels=clique_hidden_channels, out_channels=clique_hidden_channels, aggr='mean', flow=flow)
        clique2nodes_conv = partial(Clique2NodeConvBasic, in_channels=clique_hidden_channels, out_channels=node_hidden_channels)

        for _ in range(n_inter_layers-1):
            self.model.append(HMPConv(nodes_conv=nodes_conv(nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                                               nn.ReLU(),
                                                                               nn.Linear(node_hidden_channels, node_hidden_channels * node_hidden_channels)])),
                                      nodes2clique_conv=nodes2clique_conv(),
                                      clique_conv=clique_conv(nn=nn.Sequential(*[nn.Linear(clique_edge_emb_size, clique_hidden_channels),
                                                                                 nn.ReLU(),
                                                                                 nn.Linear(clique_hidden_channels, clique_hidden_channels * clique_hidden_channels)])),
                                      clique2nodes_conv=clique2nodes_conv()))

        nodes_conv = partial(NNConv, in_channels=node_hidden_channels, out_channels=node_output_channels, aggr='mean', flow=flow)
        nodes2clique_conv = partial(Node2CliqueConvBasic, in_channels=node_output_channels, out_channels=clique_hidden_channels)
        clique_conv = partial(NNConv, in_channels=clique_hidden_channels, out_channels=clique_output_channels, aggr='mean', flow=flow)
        clique2nodes_conv = partial(Clique2NodeConvBasic, in_channels=clique_output_channels, out_channels=node_output_channels)

        self.model.append(HMPConv(nodes_conv=nodes_conv(nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                                           nn.ReLU(),
                                                                           nn.Linear(node_hidden_channels, node_hidden_channels * node_output_channels)])),
                                  nodes2clique_conv=nodes2clique_conv(),
                                  clique_conv=clique_conv(nn=nn.Sequential(*[nn.Linear(clique_edge_emb_size, clique_hidden_channels),
                                                                             nn.ReLU(),
                                                                             nn.Linear(clique_hidden_channels, clique_hidden_channels * clique_output_channels)])),
                                  clique2nodes_conv=clique2nodes_conv()))

    def forward(self, node_features, edge_index, edge_features, clique_features, node2clique_index, clique_edge_index, clique_edge_features):
        for layer in self.model:
            node_features, clique_features = layer(node_features, edge_index, edge_features, clique_features, node2clique_index, clique_edge_index, clique_edge_features)
        return node_features, clique_features


class PNAConvPlainModel(torch.nn.Module):
    def __init__(self, n_inter_layers, node_hidden_channels, node_output_channels, edge_emb_size, msg_channels,
                 avg_deg_lin, avg_deg_log, flow='source_to_target'):
        super().__init__()
        self.model = nn.ModuleList()
        for _ in range(n_inter_layers - 1):
            self.model.append(PNAConv(avg_deg_lin=avg_deg_lin,
                                      avg_deg_log=avg_deg_log,
                                      in_channels=node_hidden_channels,
                                      msg_channels=msg_channels,
                                      out_channels=node_hidden_channels,
                                      flow=flow,
                                      nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                         nn.ReLU(),
                                                         nn.Linear(node_hidden_channels, node_hidden_channels * msg_channels)])))

        self.model.append(PNAConv(avg_deg_lin=avg_deg_lin,
                                  avg_deg_log=avg_deg_log,
                                  in_channels=node_hidden_channels,
                                  msg_channels=msg_channels,
                                  out_channels=node_output_channels,  # For PNAConv only this changes!
                                  flow=flow,
                                  nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                     nn.ReLU(),
                                                     nn.Linear(node_hidden_channels, node_hidden_channels * msg_channels)])))

    def forward(self, node_features, edge_index, edge_features):
        for layer in self.model:
            node_features = layer(node_features, edge_index, edge_features)
            node_features = torch.relu(node_features)
        return node_features


class PNAConvHierarchicalModel(torch.nn.Module):
    def __init__(self, n_inter_layers,
                 node_hidden_channels, node_output_channels, edge_emb_size, msg_channels,
                 clique_hidden_channels, clique_output_channels, clique_edge_emb_size,
                 avg_deg_lin, avg_deg_log, flow='source_to_target'):
        super().__init__()
        self.model = nn.ModuleList()

        # distribute complexity of reading comprehension
        nodes_conv = partial(PNAConv, avg_deg_lin=avg_deg_lin, avg_deg_log=avg_deg_log, in_channels=node_hidden_channels, msg_channels=msg_channels, out_channels=node_hidden_channels, flow=flow)
        nodes2clique_conv = partial(Node2CliqueConvBasic, in_channels=node_hidden_channels, out_channels=clique_hidden_channels)
        clique_conv = partial(NNConv, in_channels=clique_hidden_channels, out_channels=clique_hidden_channels, aggr='mean', flow=flow)
        clique2nodes_conv = partial(Clique2NodeConvBasic, in_channels=clique_hidden_channels, out_channels=node_hidden_channels)

        for _ in range(n_inter_layers-1):
            self.model.append(HMPConv(nodes_conv=nodes_conv(nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                                               nn.ReLU(),
                                                                               nn.Linear(node_hidden_channels, node_hidden_channels * msg_channels)])),
                                      nodes2clique_conv=nodes2clique_conv(),
                                      clique_conv=clique_conv(nn=nn.Sequential(*[nn.Linear(clique_edge_emb_size, clique_hidden_channels),
                                                                                 nn.ReLU(),
                                                                                 nn.Linear(clique_hidden_channels, clique_hidden_channels * clique_hidden_channels)])),
                                      clique2nodes_conv=clique2nodes_conv()))

        nodes_conv = partial(PNAConv, avg_deg_lin=avg_deg_lin, avg_deg_log=avg_deg_log, in_channels=node_hidden_channels, msg_channels=msg_channels, out_channels=node_output_channels, flow=flow)
        nodes2clique_conv = partial(Node2CliqueConvBasic, in_channels=node_output_channels, out_channels=clique_hidden_channels)
        clique_conv = partial(NNConv, in_channels=clique_hidden_channels, out_channels=clique_output_channels, aggr='mean', flow=flow)
        clique2nodes_conv = partial(Clique2NodeConvBasic, in_channels=clique_output_channels, out_channels=node_output_channels)

        self.model.append(HMPConv(nodes_conv=nodes_conv(nn=nn.Sequential(*[nn.Linear(edge_emb_size, node_hidden_channels),
                                                                           nn.ReLU(),
                                                                           nn.Linear(node_hidden_channels, node_hidden_channels * msg_channels)])),
                                  nodes2clique_conv=nodes2clique_conv(),
                                  clique_conv=clique_conv(nn=nn.Sequential(*[nn.Linear(clique_edge_emb_size, clique_hidden_channels),
                                                                             nn.ReLU(),
                                                                             nn.Linear(clique_hidden_channels, clique_hidden_channels * clique_output_channels)])),
                                  clique2nodes_conv=clique2nodes_conv()))

    def forward(self, node_features, edge_index, edge_features, clique_features, node2clique_index, clique_edge_index, clique_edge_features):
        for layer in self.model:
            node_features, clique_features = layer(node_features, edge_index, edge_features, clique_features, node2clique_index, clique_edge_index, clique_edge_features)
        return node_features, clique_features


class MPNNWithPlainNNConv(torch.nn.Module):
    def __init__(self, n_inter_layers,
                 max_z, node_emb_size, node_hidden_channels, node_output_channels,
                 edge_emb_type, max_edge_value, edge_emb_size,
                 flow='source_to_target'):
        super().__init__()
        self.node_emb = NodeEmb(max_z, node_emb_size, node_hidden_channels)
        self.edge_emb = EdgeEmb(edge_emb_type, max_edge_value, edge_emb_size)
        self.model = NNConvPlainModel(n_inter_layers, node_hidden_channels, node_output_channels, edge_emb_size, flow)
        self.pooling = GlobalPooling(hierarchical=False, node_channels=node_output_channels)

    def forward(self, graph):
        node_features = self.node_emb(graph)
        edge_features = self.edge_emb(graph)
        node_features = self.model(node_features, graph.edge_index, edge_features)
        output = self.pooling(node_features, graph.batch)
        return output


class MPNNWithHierarchicalNNConv(torch.nn.Module):
    def __init__(self, n_inter_layers,
                 max_z, node_emb_size, node_hidden_channels, node_output_channels,
                 edge_emb_type, max_edge_value, edge_emb_size,
                 max_vocab_value, clique_emb_size, clique_hidden_channels, clique_output_channels,
                 clique_edge_emb_size,
                 flow='source_to_target'):
        super().__init__()
        self.node_emb = NodeEmb(max_z, node_emb_size, node_hidden_channels)
        self.clique_emb = CliqueEmb(max_vocab_value, clique_emb_size, clique_hidden_channels)
        self.edge_emb = EdgeEmb(edge_emb_type, max_edge_value, edge_emb_size)
        self.clique_edge_emb = CliqueEdgeEmb(clique_edge_emb_size)
        self.model = NNConvHierarchicalModel(n_inter_layers, node_hidden_channels, node_output_channels, edge_emb_size, clique_hidden_channels, clique_output_channels, clique_edge_emb_size, flow)
        self.pooling = GlobalPooling(hierarchical=True, node_channels=node_output_channels, clique_channels=clique_output_channels)

    def forward(self, graph):
        node_features = self.node_emb(graph)
        clique_features = self.clique_emb(graph)
        edge_features = self.edge_emb(graph)
        clique_edge_features = self.clique_edge_emb(graph)
        node_features, clique_features = self.model(node_features, graph.edge_index, edge_features, clique_features, graph.node2clique_index, graph.clique_edge_index, clique_edge_features)
        clique_batch = torch.repeat_interleave(graph.num_cliques)
        output = self.pooling(node_features, graph.batch, clique_features, clique_batch)
        return output


class MPNNWithPlainPNAConv(torch.nn.Module):
    def __init__(self, n_inter_layers,
                 max_z, node_emb_size, node_hidden_channels, node_output_channels,
                 edge_emb_type, max_edge_value, edge_emb_size, msg_channels,
                 avg_deg_lin, avg_deg_log, flow='source_to_target'):
        super().__init__()
        self.node_emb = NodeEmb(max_z, node_emb_size, node_hidden_channels)
        self.edge_emb = EdgeEmb(edge_emb_type, max_edge_value, edge_emb_size)
        self.model = PNAConvPlainModel(n_inter_layers, node_hidden_channels, node_output_channels, edge_emb_size, msg_channels, avg_deg_lin, avg_deg_log, flow)
        self.pooling = GlobalPooling(hierarchical=False, node_channels=node_output_channels)

    def forward(self, graph):
        node_features = self.node_emb(graph)
        edge_features = self.edge_emb(graph)
        node_features = self.model(node_features, graph.edge_index, edge_features)
        output = self.pooling(node_features, graph.batch)
        return output


class MPNNWithHierarchicalPNAConv(torch.nn.Module):
    def __init__(self, n_inter_layers,
                 max_z, node_emb_size, node_hidden_channels, node_output_channels,
                 edge_emb_type, max_edge_value, edge_emb_size, msg_channels,
                 avg_deg_lin, avg_deg_log,
                 max_vocab_value, clique_emb_size, clique_hidden_channels, clique_output_channels,
                 clique_edge_emb_size,
                 flow='source_to_target'):
        super().__init__()
        self.node_emb = NodeEmb(max_z, node_emb_size, node_hidden_channels)
        self.clique_emb = CliqueEmb(max_vocab_value, clique_emb_size, clique_hidden_channels)
        self.edge_emb = EdgeEmb(edge_emb_type, max_edge_value, edge_emb_size)
        self.clique_edge_emb = CliqueEdgeEmb(clique_edge_emb_size)
        self.model = PNAConvHierarchicalModel(n_inter_layers, node_hidden_channels, node_output_channels, edge_emb_size, msg_channels, clique_hidden_channels, clique_output_channels, clique_edge_emb_size, avg_deg_lin, avg_deg_log, flow)
        self.pooling = GlobalPooling(hierarchical=True, node_channels=node_output_channels, clique_channels=clique_output_channels)

    def forward(self, graph):
        node_features = self.node_emb(graph)
        clique_features = self.clique_emb(graph)
        edge_features = self.edge_emb(graph)
        clique_edge_features = self.clique_edge_emb(graph)
        node_features, clique_features = self.model(node_features, graph.edge_index, edge_features, clique_features, graph.node2clique_index, graph.clique_edge_index, clique_edge_features)
        clique_batch = torch.repeat_interleave(graph.num_cliques)
        output = self.pooling(node_features, graph.batch, clique_features, clique_batch)
        return output
