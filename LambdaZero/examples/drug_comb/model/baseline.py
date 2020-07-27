import torch
from torch.nn import functional as F


class BaselineMLP(torch.nn.Module):
    def __init__(self, config):
        super(BaselineMLP, self).__init__()

        self.device = config["device"]

        self.num_cell_lines = config["num_cell_lines"]

        self.layer1 = torch.nn.Linear(config["in_channels"] + self.num_cell_lines, 300)
        self.layer2 = torch.nn.Linear(300, 100)

        self.activation = torch.nn.ReLU()

        self.criterion = torch.nn.MSELoss()

    def forward(self, data, conv_edges, drug_drug_batch):

        #####################################################
        # Build batch data
        #####################################################

        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        x_drug_1s = data.x[drug_1s]
        x_drug_2s = data.x[drug_2s]

        # One hot embedding for cell lines
        one_hot_cell_lines = torch.zeros((batch_size, self.num_cell_lines)).to(self.device)
        one_hot_cell_lines = one_hot_cell_lines.scatter(1, cell_lines[:, None], 1)

        batch_data_1 = torch.cat((x_drug_1s, one_hot_cell_lines.float()), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, one_hot_cell_lines.float()), dim=1)

        #####################################################
        # Forward
        #####################################################

        h_1 = self.layer2(self.activation(self.layer1(batch_data_1)))
        h_2 = self.layer2(self.activation(self.layer1(batch_data_2)))

        return h_1.reshape(batch_size, 1, -1).bmm(h_2.reshape(batch_size, -1, 1))[:, 0, 0]

    def loss(self, output, drug_drug_batch):

        # Using HSA scores for now
        ground_truth_scores = drug_drug_batch[2][:, 3]

        return self.criterion(output, ground_truth_scores)
