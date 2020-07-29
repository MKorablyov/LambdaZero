import torch
from torch.nn import functional as F


class Dummy(torch.nn.Module):
    def __init__(self, config, data):
        super(Dummy, self).__init__()

        self.param = torch.nn.Linear(1, 1)  # So that optimizer does not get an empty parameter list
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch):

        batch_size = drug_drug_batch[0].shape[0]

        return torch.zeros(batch_size, requires_grad=True) + data.ddi_edge_attr.mean()

    def loss(self, output, drug_drug_batch):

        ground_truth_scores = drug_drug_batch[2][:, 0]

        return self.criterion(output, ground_truth_scores)


class BaselineMLP(torch.nn.Module):
    def __init__(self, config, data):
        super(BaselineMLP, self).__init__()

        self.device = config["device"]
        self.cell_line = config['cell_line']

        self.layers = []

        if self.cell_line == 'all':
            self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
            self.layers.append(torch.nn.Linear(data.x_drugs.shape[1] + self.num_cell_lines, config["layers"][0]))
        else:
            self.layers.append(torch.nn.Linear(data.x_drugs.shape[1], config["layers"][0]))

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
        if self.cell_line == 'all':
            one_hot_cell_lines = torch.zeros((batch_size, self.num_cell_lines)).to(self.device)
            one_hot_cell_lines = one_hot_cell_lines.scatter(1, cell_lines[:, None], 1)

            batch_data_1 = torch.cat((x_drug_1s, one_hot_cell_lines.float()), dim=1)
            batch_data_2 = torch.cat((x_drug_2s, one_hot_cell_lines.float()), dim=1)
        else:
            batch_data_1 = x_drug_1s
            batch_data_2 = x_drug_2s

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


class SimpleBaselineFP(torch.nn.Module):
    def __init__(self, config, data):
        super(SimpleBaselineFP, self).__init__()

        self.device = config["device"]
        self.cell_line = config['cell_line']

        self.layers = []

        if self.cell_line == 'all':
            self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
            self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1] + self.num_cell_lines, config["layers"][0]))
        else:
            self.layers.append(torch.nn.Linear(2 * data.x_drugs.shape[1], config["layers"][0]))

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
        if self.cell_line == 'all':
            one_hot_cell_lines = torch.zeros((batch_size, self.num_cell_lines)).to(self.device)
            one_hot_cell_lines = one_hot_cell_lines.scatter(1, cell_lines[:, None], 1)

            batch_data_1 = torch.cat((x_drug_1s, one_hot_cell_lines.float()), dim=1)
            batch_data_2 = torch.cat((x_drug_2s, one_hot_cell_lines.float()), dim=1)
        else:
            batch_data_1 = x_drug_1s
            batch_data_2 = x_drug_2s

        batch_order_1 = torch.cat((batch_data_1, batch_data_2), dim=1)
        batch_order_2 = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch_order_1 = self.activation(self.layers[i](batch_order_1))
            batch_order_2 = self.activation(self.layers[i](batch_order_2))

        batch_order_1 = self.layers[-1](batch_order_1)
        batch_order_2 = self.layers[-1](batch_order_2)

        return batch_order_1[:, 0], batch_order_2[:, 0]

    def loss(self, output, drug_drug_batch):

        order_1, order_2 = output

        ground_truth_scores = drug_drug_batch[2][:, 0]

        return (self.criterion(order_1, ground_truth_scores) + self.criterion(order_2, ground_truth_scores)) / 2


class BaselineProt(torch.nn.Module):
    def __init__(self, config, data):
        super(BaselineProt, self).__init__()

        self.device = config["device"]
        self.cell_line = config['cell_line']

        self.layers = []

        if self.cell_line == 'all':
            self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
            self.layers.append(torch.nn.Linear(data.number_of_proteins + self.num_cell_lines, config["layers"][0]))
        else:
            self.layers.append(torch.nn.Linear(data.number_of_proteins, config["layers"][0]))

        for i in range(len(config["layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["layers"][i], config["layers"][i+1]))

        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

        self.number_of_proteins = data.number_of_proteins

        #####################################################
        # Create dictionary linking drug to targets
        #####################################################

        self.drug2target_dict = {i: [] for i in range(data.number_of_drugs)}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(int(edge[1]) - data.number_of_drugs)

    def forward(self, data, drug_drug_batch):

        #####################################################
        # Build batch data
        #####################################################

        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        prot_1 = torch.zeros((batch_size, self.number_of_proteins)).to(self.device)
        prot_2 = torch.zeros((batch_size, self.number_of_proteins)).to(self.device)

        for i in range(batch_size):
            prot_1[i, self.drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, self.drug2target_dict[int(drug_2s[i])]] = 1

        # One hot embedding for cell lines
        if self.cell_line == 'all':
            one_hot_cell_lines = torch.zeros((batch_size, self.num_cell_lines)).to(self.device)
            one_hot_cell_lines = one_hot_cell_lines.scatter(1, cell_lines[:, None], 1)

            batch_data_1 = torch.cat((prot_1, one_hot_cell_lines.float()), dim=1)
            batch_data_2 = torch.cat((prot_2, one_hot_cell_lines.float()), dim=1)
        else:
            batch_data_1 = prot_1
            batch_data_2 = prot_2

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


class SimpleBaselineProt(torch.nn.Module):
    def __init__(self, config, data):
        super(SimpleBaselineProt, self).__init__()

        self.device = config["device"]
        self.cell_line = config['cell_line']

        self.layers = []

        if self.cell_line == 'all':
            self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
            self.layers.append(torch.nn.Linear(2 * data.number_of_proteins + self.num_cell_lines, config["layers"][0]))
        else:
            self.layers.append(torch.nn.Linear(2 * data.number_of_proteins, config["layers"][0]))

        for i in range(len(config["layers"]) - 1):
            self.layers.append(torch.nn.Linear(config["layers"][i], config["layers"][i+1]))

        self.layers = torch.nn.ModuleList(self.layers)

        self.activation = torch.nn.ReLU()
        self.criterion = torch.nn.MSELoss()

        self.number_of_proteins = data.number_of_proteins

        #####################################################
        # Create dictionary linking drug to targets
        #####################################################

        self.drug2target_dict = {i: [] for i in range(data.number_of_drugs)}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(int(edge[1]) - data.number_of_drugs)

    def forward(self, data, drug_drug_batch):

        #####################################################
        # Build batch data
        #####################################################

        batch_size = drug_drug_batch[0].shape[0]
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        prot_1 = torch.zeros((batch_size, self.number_of_proteins)).to(self.device)
        prot_2 = torch.zeros((batch_size, self.number_of_proteins)).to(self.device)

        for i in range(batch_size):
            prot_1[i, self.drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, self.drug2target_dict[int(drug_2s[i])]] = 1

        # One hot embedding for cell lines
        if self.cell_line == 'all':
            one_hot_cell_lines = torch.zeros((batch_size, self.num_cell_lines)).to(self.device)
            one_hot_cell_lines = one_hot_cell_lines.scatter(1, cell_lines[:, None], 1)

            batch_data_1 = torch.cat((prot_1, one_hot_cell_lines.float()), dim=1)
            batch_data_2 = torch.cat((prot_2, one_hot_cell_lines.float()), dim=1)
        else:
            batch_data_1 = prot_1
            batch_data_2 = prot_2

        batch_order_1 = torch.cat((batch_data_1, batch_data_2), dim=1)
        batch_order_2 = torch.cat((batch_data_2, batch_data_1), dim=1)

        #####################################################
        # Forward
        #####################################################

        for i in range(len(self.layers) - 1):
            batch_order_1 = self.activation(self.layers[i](batch_order_1))
            batch_order_2 = self.activation(self.layers[i](batch_order_2))

        batch_order_1 = self.layers[-1](batch_order_1)
        batch_order_2 = self.layers[-1](batch_order_2)

        return batch_order_1[:, 0], batch_order_2[:, 0]

    def loss(self, output, drug_drug_batch):

        order_1, order_2 = output

        ground_truth_scores = drug_drug_batch[2][:, 0]

        return (self.criterion(order_1, ground_truth_scores) + self.criterion(order_2, ground_truth_scores)) / 2