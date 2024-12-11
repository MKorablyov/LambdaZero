from LambdaZero.examples.drug_comb.dataset.drug_combdb_data import DrugCombDb
import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Parameter


class GCN(torch.nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(config["in_channels"], 16, cached=True)
        self.conv2 = GCNConv(16, 16, cached=True)

        self.predictor = Parameter(torch.randn((16, 16)))

        self.criterion = torch.nn.MSELoss()

    def forward(self, data, conv_edges):
        x = self.conv1(data.x, conv_edges)
        x = F.relu(x)
        x = self.conv2(x, conv_edges)

        return x[:data.number_of_drugs].mm(self.predictor).mm(x[:data.number_of_drugs].T)

    def loss(self, output, drug_drug_batch, number_of_drugs):

        # Keep only relevant predictions
        mask = torch.sparse.FloatTensor(drug_drug_batch[0].T,
                                        torch.ones(drug_drug_batch[0].shape[0]),
                                        torch.Size([number_of_drugs, number_of_drugs])).to_dense()

        # Using Zip scores for now
        scores = torch.sparse.FloatTensor(drug_drug_batch[0].T,
                                          drug_drug_batch[2][:, 3],
                                          torch.Size([number_of_drugs, number_of_drugs])).to_dense()

        output *= mask

        return ((output - scores)**2).sum()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset
    dataset = DrugCombDb()
    # restrict ourselves to fingerprint features:
    dataset.data.x = dataset.data.x[:, :1024]

    # Send data to device
    data = dataset[0].to(device)

    # Initialize model
    model = GCN(dataset.data.x.shape[1])

    # Feed forward
    model(dataset.data)
