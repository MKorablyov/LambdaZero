from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch

class SubgraphEmbeddingRegressorModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conv1 = GCNConv(config["in_channels"], config["embed_channels"])
        self.conv2 = GCNConv(config["embed_channels"], config["embed_channels"])

        # The input to the regressor will be the concatenation of two graph
        # embeddings, so take the in channels here to be 2 times the embedding size
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(2 * config["embed_channels"], config["regressor_hidden_channels"]),
            torch.nn.ReLU(),
            torch.nn.Linear(config["regressor_hidden_channels"], config["out_channels"])
        )

    def forward(self, drug_drug_batch, subgraph_batch):
        import pdb; pdb.set_trace()
        x =  subgraph_batch.x
        for conv in [self.conv1, self.conv2]:
            x = conv(x, subgraph_batch.edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.training)

        node_embeds = x
        graph_embeds = scatter_mean(node_embeds)

        # Quantize drug drug batch so indices match graph_embeds
        drug_bins = np.unique(drug_drug_batch.flatten()) + 1
        drug_drug_batch_qtzd = torch.from_numpy(np.digitize(drug_drug_batch, drug_bins))

        from_drug_embeds = graph_embeds[drug_drug_batch_qtzd[0,:]]
        to_drug_embeds = graph_embeds[drug_drug_batch_qtzd[1,:]]

        concatenated_embed_pairs = np.hstack([from_drug_embeds, to_drug_embeds])

        return self.regressor(concatenated_embed_pairs)

