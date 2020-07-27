import torch
from torch.nn import functional as F


class BaselineMLP(torch.nn.Module):
    def __init__(self, config, data):
        super(BaselineMLP, self).__init__()

        self.device = config["device"]

        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))

        self.layers = []
        self.layers.append(torch.nn.Linear(data.x_drugs.shape[1] + self.num_cell_lines, config["layers"][0]))

        for i in range(len(config["layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["layers"][i], config["layers"][i+1]))

        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch):

        #####################################################
        # Build batch data
        #####################################################

        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        # One hot embedding for cell lines
        one_hot_cell_lines = torch.zeros((batch_size, self.num_cell_lines)).to(self.device)
        one_hot_cell_lines = one_hot_cell_lines.scatter(1, cell_lines[:, None], 1)

        batch_data_1 = torch.cat((x_drug_1s, one_hot_cell_lines.float()), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, one_hot_cell_lines.float()), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch_data_1 = self.activation(self.layers[i](batch_data_1))
            batch_data_2 = self.activation(self.layers[i](batch_data_2))

        batch_data_1 = self.layers[-1](batch_data_1)
        batch_data_2 = self.layers[-1](batch_data_2)

        return batch_data_1.reshape(batch_size, 1, -1).bmm(batch_data_2.reshape(batch_size, -1, 1))[:, 0, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[2][:, 0]

        return self.criterion(output, ground_truth_scores)
