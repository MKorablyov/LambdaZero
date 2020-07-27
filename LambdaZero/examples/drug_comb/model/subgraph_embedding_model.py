from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean
from collections import defaultdict
import torch.nn.functional as F
import numpy as np
import torch
import random

class SubgraphEmbeddingRegressorModel(torch.nn.Module):
    def __init__(self, config, cell_lines):
        super().__init__()

        self.out_channels = config["out_channels"]
        self.conv1 = GCNConv(config["in_channels"], config["embed_channels"])
        self.conv2 = GCNConv(config["embed_channels"], config["embed_channels"])

        self.conv_dropout = torch.nn.Dropout(config['conv_dropout_rate'])
        self.linear_dropout = torch.nn.Dropout(config['linear_dropout_rate'])

        if config['prediction_type'] == 'mlp':
            self.shared_lin_lyr = torch.nn.Linear(2 * config["embed_channels"],
                                                  config["regressor_hidden_channels"])

            # The input to the regressor will be the concatenation of two graph
            # embeddings, so take the in channels here to be 2 times the embedding size
            self.cell_line_to_regressor = {
                cell_line.item(): torch.nn.Linear(config["regressor_hidden_channels"], config["out_channels"])
                for cell_line in cell_lines
            }

        final_predictors = {
            'mlp': self._pred_mlp,
            'dot_product': self._pred_dot_product,
        }

        self._predict_with_edge_embeds = final_predictors[config['prediction_type']]

    def to(self, device):
        new_model = super().to(device)

        if hasattr(new_model, 'cell_line_to_regressor'):
            for cell_line, regressor in new_model.cell_line_to_regressor.items():
                new_model.cell_line_to_regressor[cell_line] = regressor.to(device)

        return new_model

    def forward(self, x, drug_drug_batch, edge_cell_lines, sg_edge_index, sg_nodes, sg_avging_idx):
        x = F.relu(self.conv1(x, sg_edge_index))
        x = self.conv_dropout(x)
        x = F.relu(self.conv2(x, sg_edge_index))

        node_embeds = x[sg_nodes]
        graph_embeds = scatter_mean(node_embeds, sg_avging_idx, dim=0)

        return self._pred_with_graph_embeds(graph_embeds, drug_drug_batch, edge_cell_lines)

    def _pred_with_graph_embeds(self, graph_embeds, drug_drug_batch, edge_cell_lines):
        # Quantize drug drug batch so indices match graph_embeds
        drug_bins = np.unique(drug_drug_batch.cpu().flatten()) + 1
        drug_drug_batch = torch.from_numpy(np.digitize(drug_drug_batch.cpu(), drug_bins))

        from_drug_embeds = graph_embeds[drug_drug_batch[0,:]]
        to_drug_embeds = graph_embeds[drug_drug_batch[1,:]]

        return self._predict_with_edge_embeds(from_drug_embeds, to_drug_embeds, edge_cell_lines)

    def _pred_mlp(self, from_drug_embeds, to_drug_embeds, edge_cell_lines):
        x = None
        if bool(random.getrandbits(1)):
            x = torch.cat((from_drug_embeds, to_drug_embeds), dim=1)
        else:
            x = torch.cat((to_drug_embeds, from_drug_embeds), dim=1)

        x = self.shared_lin_lyr(x)
        x = self.linear_dropout(x)

        cell_line_to_idx = defaultdict(list)
        for i, cell_line in enumerate(edge_cell_lines):
            cell_line_to_idx[cell_line.item()].append(i)

        preds = torch.empty((from_drug_embeds.shape[1], self.out_channels),
                            device=from_drug_embeds.device)
        for cell_line, cell_line_idxs in cell_line_to_idx.items():
            regressor = self.cell_line_to_regressor[cell_line]
            preds[cell_line_idxs] = regressor(x[cell_line_idxs])

        return preds

    def _pred_dot_product(self, from_drug_embeds, to_drug_embeds, edge_cell_lines):
        return torch.sum(from_drug_embeds * to_drug_embeds, dim=1)

