import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter, ParameterList
import numpy as np


########################################################################################################################
# Modules
########################################################################################################################


class ResidualModule(torch.nn.Module):
    def __init__(self, ConvLayer, drug_channels, prot_channels, data):
        super(ResidualModule, self).__init__()
        self.conv1 = ConvLayer(drug_channels, prot_channels, drug_channels, prot_channels, data)
        self.conv2 = ConvLayer(drug_channels, prot_channels, drug_channels, prot_channels, data)

    def forward(self, h_drug, h_prot, data):
        out_drug, out_prot = self.conv1(h_drug, h_prot, data)
        out_drug = F.relu(out_drug)
        out_prot = F.relu(out_prot)
        out_drug, out_prot = self.conv2(out_drug, out_prot, data)

        return F.relu(h_drug + out_drug), F.relu(h_prot + out_prot)


class LowRankAttention(torch.nn.Module):
    def __init__(self, k, d, dropout):
        super().__init__()
        self.w = torch.nn.Sequential(torch.nn.Linear(d, 4*k), torch.nn.ReLU())
        self.activation = torch.nn.ReLU()
        self.apply(weight_init)
        self.k = k
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, X):
        tmp = self.w(X)
        U = tmp[:, :self.k]
        V = tmp[:, self.k:2*self.k]
        Z = tmp[:, 2*self.k:3*self.k]
        T = tmp[:, 3*self.k:]
        V_T = torch.t(V)
        # normalization
        D = joint_normalize2(U, V_T)
        res = torch.mm(U, torch.mm(V_T, Z))
        res = torch.cat((res*D, T), dim=1)
        return self.dropout(res)


def joint_normalize2(U, V_T):
    # U and V_T are in block diagonal form
    if torch.cuda.is_available():
        tmp_ones = torch.ones((V_T.shape[1], 1)).to("cuda")
    else:
        tmp_ones = torch.ones((V_T.shape[1], 1))
    norm_factor = torch.mm(U, torch.mm(V_T, tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1/norm_factor


def weight_init(layer):
    if type(layer) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight.data)
        # nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data,0)
    return


########################################################################################################################
# Simple Predictors
########################################################################################################################


class InnerProductPredictor(torch.nn.Module):
    def __init__(self, data, num_cell_lines, layer_dims):
        """
        Non linear transformation (linear layer + Relu) applied to each drug embedding independently
        Then the score is predited as <h_1 , G . h_2> where G is learnt. We enforce G symmetric

        All transformations are cell line specific
        """
        super(InnerProductPredictor, self).__init__()
        assert len(layer_dims) == 2

        self.predictor_lin = Parameter(1 / 100 * torch.randn((num_cell_lines, layer_dims[0],
                                                              layer_dims[1])))
        self.predictor_prod = Parameter(1 / 100 * torch.randn((num_cell_lines, layer_dims[1],
                                                               layer_dims[1])))  # Mat for outer product

    def forward(self, data, drug_drug_batch, h_drug):
        batch_size = drug_drug_batch[0].shape[0]

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1s = h_drug[drug_1s].reshape(batch_size, 1, -1)  # Trick to allow broadcasting with matmul
        h_drug_2s = h_drug[drug_2s].reshape(batch_size, 1, -1)  # Trick to allow broadcasting with matmul
        # Cell line specific linear layer
        h_drug_1s = F.relu(h_drug_1s.matmul(self.predictor_lin[cell_lines]))
        h_drug_2s = F.relu(h_drug_2s.matmul(self.predictor_lin[cell_lines]))

        h_drug_2s = h_drug_2s.reshape(batch_size, -1, 1)  # Trick to allow broadcasting with matmul

        # Build G to be symmetric
        G = self.predictor_prod[cell_lines] + self.predictor_prod[cell_lines].permute(0, 2, 1)

        # Inner product of embedding pairs using G
        batch_score_preds = h_drug_1s.matmul(G).matmul(h_drug_2s)[:, 0, 0]

        return batch_score_preds


class MLPPredictor(torch.nn.Module):
    """
    Does not take into account cell line
    """
    def __init__(self, data, num_cell_lines, layer_dims):
        super(MLPPredictor, self).__init__()
        assert layer_dims[-1] == 1

        layers = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i != len(layer_dims) - 2:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, data, drug_drug_batch, h_drug):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch

        h_drug_1s = h_drug[drug_1s]
        h_drug_2s = h_drug[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((h_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((h_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        batch_score_preds = self.mlp(batch)[:, 0]

        return batch_score_preds

########################################################################################################################
# Predictors with conditioning
########################################################################################################################


class FilmMLPPredictor(torch.nn.Module):
    """
    Takes into account cell line with Film conditioning. Takes fingerprints as input as well
    """

    def __init__(self, data, num_cell_lines, layer_dims):
        super(FilmMLPPredictor, self).__init__()
        assert layer_dims[-1] == 1

        layers = []
        film = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            film.append(Parameter(1/100 * torch.randn(num_cell_lines, 2) + torch.Tensor([[1, 0]])))

        self.layers = nn.ModuleList(layers)
        self.film = nn.ParameterList(film)

    def forward(self, data, drug_drug_batch, h_drug):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1s = h_drug[drug_1s]
        h_drug_2s = h_drug[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((h_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((h_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        for i in range(len(self.layers)):
            batch = self.layers[i](batch)
            if i != len(self.layers) - 1:
                # Film conditioning
                batch = self.film[i][cell_lines][:, :1] * batch + self.film[i][cell_lines][:, 1:]
                batch = F.relu(batch)

        batch_score_preds = batch[:, 0]

        return batch_score_preds


class CellLineMLPPredictor(torch.nn.Module):
    """
    Takes into account cell line. All layers but the last two are shared between
    cell lines. The last two layers are cell line specific
    The forward passes are all computed in parallel
    """

    def __init__(self, data, num_cell_lines, layer_dims):
        super(CellLineMLPPredictor, self).__init__()
        assert layer_dims[-1] == 1

        last_layers = []
        shared_layers = []
        offsets = []

        # Shared layers
        for i in range(len(layer_dims) - 3):
            shared_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            shared_layers.append(nn.ReLU())

        # Cell line specific layers
        for i in range(len(layer_dims) - 3, len(layer_dims) - 1):
            last_layers.append(Parameter(1 / 100 * torch.randn((num_cell_lines, layer_dims[i + 1], layer_dims[i]))))
            offsets.append(Parameter(1 / 100 * torch.randn((num_cell_lines, layer_dims[i + 1], 1))))

        self.shared_layers = nn.ModuleList(shared_layers)
        self.last_layers = ParameterList(last_layers)
        self.offsets = ParameterList(offsets)

    def forward(self, data, drug_drug_batch, h_drug):
        batch_size = drug_drug_batch[0].shape[0]

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        h_drug_1s = h_drug[drug_1s]
        h_drug_2s = h_drug[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((h_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((h_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        # Forward in shared layers
        for i in range(len(self.shared_layers)):
            batch = self.shared_layers[i](batch)
            batch = F.relu(batch)

        batch = batch.reshape(batch_size, -1, 1)  # Trick to allow broadcasting with matmul

        # Forward in cell_line specific MLPs in parallel
        for i in range(len(self.last_layers)):
            batch = self.last_layers[i][cell_lines].matmul(batch) + self.offsets[i][cell_lines]
            if i != len(self.last_layers) - 1:
                batch = F.relu(batch)

        batch_score_preds = batch[:, 0, 0]

        return batch_score_preds


########################################################################################################################
# Predictors that take fingerprints as input as well
########################################################################################################################


class FingerprintMLPPredictor(torch.nn.Module):
    """
    Does not take into account cell line. Takes fingerprints as input as well
    """

    def __init__(self, data, num_cell_lines, layer_dims):
        super(FingerprintMLPPredictor, self).__init__()
        assert layer_dims[-1] == 1

        layers = []
        layer_dims[0] += data.x_drugs.shape[1] * 2

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            if i != len(layer_dims) - 2:
                layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, data, drug_drug_batch, h_drug):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        h_drug_1s = h_drug[drug_1s]
        h_drug_2s = h_drug[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((x_drug_1s, h_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, h_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        batch_score_preds = self.mlp(batch)[:, 0]

        return batch_score_preds


class FilmFingerprintMLPPredictor(torch.nn.Module):
    """
    Takes into account cell line with Film conditioning. Takes fingerprints as input as well
    """

    def __init__(self, data, num_cell_lines, layer_dims):
        super(FilmFingerprintMLPPredictor, self).__init__()
        assert layer_dims[-1] == 1

        layers = []
        layer_dims[0] += data.x_drugs.shape[1] * 2
        film = []

        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            film.append(Parameter(1/100 * torch.randn(num_cell_lines, 2) + torch.Tensor([[1, 0]])))

        self.layers = nn.ModuleList(layers)
        self.film = nn.ParameterList(film)

    def forward(self, data, drug_drug_batch, h_drug):
        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        h_drug_1s = h_drug[drug_1s]
        h_drug_2s = h_drug[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((x_drug_1s, h_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, h_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        for i in range(len(self.layers)):
            batch = self.layers[i](batch)
            if i != len(self.layers) - 1:
                # Film conditioning
                batch = self.film[i][cell_lines][:, :1] * batch + self.film[i][cell_lines][:, 1:]
                batch = F.relu(batch)

        batch_score_preds = batch[:, 0]

        return batch_score_preds


class FPCellLineMLPPredictor(torch.nn.Module):
    """
    Takes into account cell line. Takes fingerprints as input as well. All layers but the last two are shared between
    cell lines. The last two layers are cell line specific
    The forward passes are all computed in parallel
    """

    def __init__(self, data, num_cell_lines, layer_dims):
        super(FPCellLineMLPPredictor, self).__init__()
        assert layer_dims[-1] == 1

        last_layers = []
        shared_layers = []
        layer_dims[0] += data.x_drugs.shape[1] * 2
        offsets = []

        # Shared layers
        for i in range(len(layer_dims) - 3):
            shared_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            shared_layers.append(nn.ReLU())

        # Cell line specific layers
        for i in range(len(layer_dims) - 3, len(layer_dims) - 1):
            last_layers.append(Parameter(1 / 100 * torch.randn((num_cell_lines, layer_dims[i+1], layer_dims[i]))))
            offsets.append(Parameter(1 / 100 * torch.randn((num_cell_lines, layer_dims[i+1], 1))))

        self.shared_layers = nn.ModuleList(shared_layers)
        self.last_layers = ParameterList(last_layers)
        self.offsets = ParameterList(offsets)

    def forward(self, data, drug_drug_batch, h_drug):
        batch_size = drug_drug_batch[0].shape[0]

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]

        h_drug_1s = h_drug[drug_1s]
        h_drug_2s = h_drug[drug_2s]

        n_attr = drug_drug_batch[2].shape[1] // 2
        # Concatenate with IC50s
        batch_data_1 = torch.cat((x_drug_1s, h_drug_1s, drug_drug_batch[2][:, n_attr-1][:, None]), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, h_drug_2s, drug_drug_batch[2][:, -1][:, None]), dim=1)

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((batch_data_1, batch_data_2), dim=1)
        else:
            batch = torch.cat((batch_data_2, batch_data_1), dim=1)

        # Forward in shared layers
        for i in range(len(self.shared_layers)):
            batch = self.shared_layers[i](batch)
            batch = F.relu(batch)

        batch = batch.reshape(batch_size, -1, 1)  # Trick to allow broadcasting with matmul

        # Forward in cell_line specific MLPs in parallel
        for i in range(len(self.last_layers)):
            batch = self.last_layers[i][cell_lines].matmul(batch) + self.offsets[i][cell_lines]
            if i != len(self.last_layers) - 1:
                batch = F.relu(batch)

        batch_score_preds = batch[:, 0, 0]

        return batch_score_preds
