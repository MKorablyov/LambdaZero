import torch
from torch.nn import Parameter, ParameterList
from torch.nn import functional as F
import numpy as np


class Dummy(torch.nn.Module):
    def __init__(self, data, config):
        super(Dummy, self).__init__()

        self.device = config["device"]

        self.param = torch.nn.Linear(1, 1)  # So that optimizer does not get an empty parameter list
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch):

        batch_size = drug_drug_batch[0].shape[0]

        return torch.zeros(batch_size, requires_grad=True) + data.ddi_edge_attr[:, 0].mean()

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output.to(self.device), ground_truth_scores)


class ConcentrationOnlyBaseline(torch.nn.Module):
    def __init__(self, data, config):
        super(ConcentrationOnlyBaseline, self).__init__()

        self.layers = []

        self.layers.append(torch.nn.Linear(2, config["mlp_layers"][0]))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))

        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch):

        #####################################################
        # Build batch data
        #####################################################

        n_attr = drug_drug_batch[2].shape[1] // 2

        conc_1 = drug_drug_batch[2][:, n_attr-1][:, None]
        conc_2 = drug_drug_batch[2][:, -1][:, None]

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((conc_1, conc_2), dim=1)
        else:
            batch = torch.cat((conc_2, conc_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class ScalarProdBaselineFP(torch.nn.Module):
    def __init__(self, data, config):
        super(ScalarProdBaselineFP, self).__init__()

        self.device = config["device"]

        self.layers = []

        self.layers.append(torch.nn.Linear(data.x_drugs.shape[1] + 1, config["scalar_layers"][0]))

        for i in range(len(config["scalar_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["scalar_layers"][i], config["scalar_layers"][i+1]))

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

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2

        batch_data_1 = torch.cat((x_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

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

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class MLPBaselineFP(torch.nn.Module):
    def __init__(self, data, config):
        super(MLPBaselineFP, self).__init__()

        self.device = config["device"]

        self.layers = []

        self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1] + 2, config["mlp_layers"][0]))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))

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

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((x_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class MLPBaselineExpr(torch.nn.Module):
    def __init__(self, data, config):
        super(MLPBaselineExpr, self).__init__()

        self.device = config["device"]

        self.layers = []

        self.layers.append(torch.nn.Linear(data.ddi_edge_attr.shape[1], config["mlp_layers"][0]))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))

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

        n_attr = drug_drug_batch[2].shape[1] // 2

        batch_data_1 = drug_drug_batch[2][:, :n_attr]
        batch_data_2 = drug_drug_batch[2][:, n_attr:]

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class MLPBaselineFPExpr(torch.nn.Module):
    def __init__(self, data, config):
        super(MLPBaselineFPExpr, self).__init__()

        self.device = config["device"]

        self.layers = []

        self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1] + data.ddi_edge_attr.shape[1],
                                           config["mlp_layers"][0]))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))

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

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with expression
        batch_data_1 = torch.cat((x_drug_1s, drug_drug_batch[2][:, :n_attr]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, drug_drug_batch[2][:, n_attr:]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class FilmMLPBaselineFP(torch.nn.Module):
    def __init__(self, data, config):
        super(FilmMLPBaselineFP, self).__init__()
        self.device = config["device"]

        self.layers = []
        self.film = []
        num_cell_lines = len(torch.unique(data.ddi_edge_classes))

        self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1] + 2,
                                           config["mlp_layers"][0]))
        self.film.append(Parameter(1 / 100 * torch.randn(num_cell_lines, 2) + torch.Tensor([[1, 0]])))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))
            self.film.append(Parameter(1 / 100 * torch.randn(num_cell_lines, 2) + torch.Tensor([[1, 0]])))

        self.layers = torch.nn.ModuleList(self.layers)
        self.film = torch.nn.ParameterList(self.film)

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

        n_attr = drug_drug_batch[2].shape[1] // 2

        batch_data_1 = torch.cat((x_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))
            # Film conditioning
            batch = self.film[i][cell_lines][:, :1] * batch + self.film[i][cell_lines][:, 1:]
            batch = F.relu(batch)

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class MLPBaselineFPProt(torch.nn.Module):
    def __init__(self, data, config):
        super(MLPBaselineFPProt, self).__init__()
        self.device = config["device"]

        self.layers = []

        self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1] + 2 * data.x_prots.shape[0] + 2,
                                           config["mlp_layers"][0]))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))

        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

        #####################################################
        # Create dictionary linking drug to targets
        #####################################################

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(int(edge[1]) - data.x_drugs.shape[0])

    def forward(self, data, drug_drug_batch):

        batch_size = drug_drug_batch[0].shape[0]
        prot_1 = torch.zeros((batch_size, data.x_prots.shape[0])).to(self.device)
        prot_2 = torch.zeros((batch_size, data.x_prots.shape[0])).to(self.device)

        #####################################################
        # Build batch data
        #####################################################

        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        for i in range(batch_size):
            prot_1[i, self.drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, self.drug2target_dict[int(drug_2s[i])]] = 1

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2

        # Concatenate with IC50s
        batch_data_1 = torch.cat((x_drug_1s, prot_1, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, prot_2, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)


class FilmMLPBaselineFPProt(torch.nn.Module):
    def __init__(self, data, config):
        super(FilmMLPBaselineFPProt, self).__init__()
        self.device = config["device"]

        self.layers = []
        self.film = []
        num_cell_lines = len(torch.unique(data.ddi_edge_classes))

        self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1] + 2 * data.x_prots.shape[0] + 2,
                                           config["mlp_layers"][0]))
        self.film.append(Parameter(1 / 100 * torch.randn(num_cell_lines, 2) + torch.Tensor([[1, 0]])))

        for i in range(len(config["mlp_layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["mlp_layers"][i], config["mlp_layers"][i+1]))
            self.film.append(Parameter(1 / 100 * torch.randn(num_cell_lines, 2) + torch.Tensor([[1, 0]])))

        self.layers = torch.nn.ModuleList(self.layers)
        self.film = torch.nn.ParameterList(self.film)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

        #####################################################
        # Create dictionary linking drug to targets
        #####################################################

        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(int(edge[1]) - data.x_drugs.shape[0])

    def forward(self, data, drug_drug_batch):

        batch_size = drug_drug_batch[0].shape[0]
        prot_1 = torch.zeros((batch_size, data.x_prots.shape[0])).to(self.device)
        prot_2 = torch.zeros((batch_size, data.x_prots.shape[0])).to(self.device)

        #####################################################
        # Build batch data
        #####################################################

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        for i in range(batch_size):
            prot_1[i, self.drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, self.drug2target_dict[int(drug_2s[i])]] = 1

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2

        batch_data_1 = torch.cat((x_drug_1s, prot_1, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, prot_2, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch = self.activation(self.layers[i](batch))
            # Film conditioning
            batch = self.film[i][cell_lines][:, :1] * batch + self.film[i][cell_lines][:, 1:]
            batch = F.relu(batch)

        batch = self.layers[-1](batch)

        return batch[:, 0]

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)