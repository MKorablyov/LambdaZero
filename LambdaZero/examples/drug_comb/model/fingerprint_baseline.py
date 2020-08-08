import torch
from torch.nn import functional as F

class ResponseBaselineFP(torch.nn.Module):
    def __init__(self, config, fp_len):
        super(ResponseBaselineFP, self).__init__()

        self.device = config["device"]

        self.layers = []

        self.layers.append(torch.nn.Linear(fp_len + 1, config["layers"][0]))

        for i in range(len(config["layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["layers"][i], config["layers"][i+1]))

        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

    def forward(self, fingerprints, drug_drug_batch):

        #####################################################
        # Build batch data
        #####################################################

        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[1]  # Edge-head drugs in the batch

        x_drug_1s = fingerprints[drug_1s]
        x_drug_2s = fingerprints[drug_2s]

        batch_data_1 = torch.cat((x_drug_1s, drug_drug_batch[2].reshape(-1, 1)), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, drug_drug_batch[3].reshape(-1, 1)), dim=1)

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
        ground_truth_scores = drug_drug_batch[4]

        return self.criterion(output, ground_truth_scores)

