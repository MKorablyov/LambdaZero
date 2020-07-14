from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch

class SubgraphEmbeddingRegressorModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.out_channels = config["out_channels"]
        self.conv1 = GCNConv(config["in_channels"], config["embed_channels"])
        self.conv2 = GCNConv(config["embed_channels"], config["embed_channels"])

        # The input to the regressor will be the concatenation of two graph
        # embeddings, so take the in channels here to be 2 times the embedding size
        self.cell_line_to_regressor = {
            cell_line: torch.nn.Sequential(
                torch.nn.Linear(2 * config["embed_channels"], config["regressor_hidden_channels"]),
                torch.nn.ReLU(),
                torch.nn.Linear(config["regressor_hidden_channels"], config["out_channels"])
            )
            for cell_line in range(config["num_cell_lines"])
        }

        self.prediction_type = config['prediction_type']
        if not self.prediction_type in ['dot_product', 'mlp']:
            raise ValueError('prediction_type must be one of \'dot_product\' or \'mlp\'')

        if config['weight_initialization_type'] == 'xavier':
            lin_layers = [
                lyr
                for regressor in self.cell_line_to_regressor.values()
                for lyr in regressor.modules() if isinstance(lyr, torch.nn.Linear)
            ]

            for layer in [self.conv1, self.conv2] + lin_layers:
                torch.nn.init.xavier_uniform_(layer.weight)

    def to(self, device):
        new_model = super().to(device)

        for cell_line, regressor in new_model.cell_line_to_regressor.items():
            new_model.cell_line_to_regressor[cell_line] = regressor.to(device)

        return new_model

    def forward(self, drug_drug_batch, drug_protein_graph, edge_cell_lines):
        x = drug_protein_graph.x
        for conv in [self.conv1, self.conv2]:
            x = conv(x, drug_protein_graph.edge_index)
            x = F.relu(x)

        # drug_protein_graph has drug -> protein and protein -> drug edges, but for
        # graph averaging we only want the proteins for the drug -> protein edges.
        # Since the first half of edges in drug_protein_graph are drug -> protein
        # edges by construction, we take the first half here priorr to calling scatter_mean.
        num_dpi_edges = drug_protein_graph.edge_index.shape[1] // 2
        averaging_index = drug_protein_graph.edge_index[:, :num_dpi_edges]
        averaging_index = torch.sort(averaging_index, dim=0)[0]

        node_embeds = x[averaging_index[1,:]]
        graph_embeds = scatter_mean(node_embeds, averaging_index[0,:], dim=0)

        from_drug_embeds = graph_embeds[drug_drug_batch[0,:]]
        to_drug_embeds = graph_embeds[drug_drug_batch[1,:]]

        preds = None
        if self.prediction_type == 'mlp':
            concatenated_embed_pairs = torch.cat((from_drug_embeds, to_drug_embeds), dim=1)

            cell_line_to_idx = defaultdict(list)
            for i, cell_line in enumerate(edge_cell_lines):
                cell_line_to_idx[cell_line.item()].append(i)

            preds = torch.empty((drug_drug_batch.shape[1], self.out_channels),
                                device=concatenated_embed_pairs.device)
            for cell_line, cell_line_idxs in cell_line_to_idx.items():
                regressor = self.cell_line_to_regressor[cell_line]
                preds[cell_line_idxs] = regressor(concatenated_embed_pairs[cell_line_idxs])

        else:
            preds = torch.dot(from_drug_embeds, to_drug_embeds)

        return preds

