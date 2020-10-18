import torch
from torch.nn import functional as F
import numpy as np


########################################################################################################################
# Modules
########################################################################################################################


class ResidualModule(torch.nn.Module):
    def __init__(self, ConvLayer, drug_channels, prot_channels,
                 pass_d2d_msg, pass_d2p_msg, pass_p2d_msg, pass_p2p_msg, data):

        super(ResidualModule, self).__init__()
        self.conv1 = ConvLayer(drug_channels, prot_channels, drug_channels,
                               prot_channels, pass_d2d_msg, pass_d2p_msg,
                               pass_p2d_msg, pass_p2p_msg, data)

        self.conv2 = ConvLayer(drug_channels, prot_channels, drug_channels,
                               prot_channels, pass_d2d_msg, pass_d2p_msg,
                               pass_p2d_msg, pass_p2p_msg, data)

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
# Functions for Predictors
########################################################################################################################


def get_batch(data, drug_drug_batch, h_drug, drug2target_dict, with_fp=False, with_expr=False, with_prot=False):
    """
    if h_drug is None, the fingerprints are taken as embeddings of the drugs, and the parameter with_fp is ignored
    """

    batch_size = drug_drug_batch[0].shape[0]

    drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
    drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
    cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

    #####################################################
    # Get drug embeddings
    #####################################################

    if h_drug is not None:
        h_drug_1s = h_drug[drug_1s]  # Embeddings of tail drugs in the batch
        h_drug_2s = h_drug[drug_2s]  # Embeddings of head drugs in the batch
    else:  # The embedding of the drug is the fingerprint
        h_drug_1s = data.x_drugs[drug_1s]
        h_drug_2s = data.x_drugs[drug_2s]

    n_attr = drug_drug_batch[2].shape[1] // 2  # Number of dd-edge attributes

    #####################################################
    # Get drug pair attributes
    #####################################################

    if with_expr:  # Include gene expression data
        attr_1s = drug_drug_batch[2][:, :n_attr]
        attr_2s = drug_drug_batch[2][:, n_attr:]
    else:
        attr_1s = drug_drug_batch[2][:, n_attr - 1][:, None]
        attr_2s = drug_drug_batch[2][:, -1][:, None]

    #####################################################
    # Add protein target information
    #####################################################

    if with_prot:  # Include protein target information
        prot_1 = torch.zeros((batch_size, data.x_prots.shape[0]))
        prot_2 = torch.zeros((batch_size, data.x_prots.shape[0]))

        if torch.cuda.is_available():
            prot_1 = prot_1.to("cuda")
            prot_2 = prot_2.to("cuda")

        for i in range(batch_size):
            prot_1[i, drug2target_dict[int(drug_1s[i])]] = 1
            prot_2[i, drug2target_dict[int(drug_2s[i])]] = 1

        attr_1s = torch.cat((attr_1s, prot_1), dim=1)
        attr_2s = torch.cat((attr_2s, prot_2), dim=1)

    #####################################################
    # Include fingerprints
    #####################################################

    if with_fp and h_drug is not None:  # Concatenate the embeddings with fingerprints and IC50s
        x_drug_1s = data.x_drugs[drug_1s]
        x_drug_2s = data.x_drugs[drug_2s]
        batch_data_1 = torch.cat((x_drug_1s, h_drug_1s, attr_1s), dim=1)
        batch_data_2 = torch.cat((x_drug_2s, h_drug_2s, attr_2s), dim=1)
    else:  # Concatenate with IC50 only
        batch_data_1 = torch.cat((h_drug_1s, attr_1s), dim=1)
        batch_data_2 = torch.cat((h_drug_2s, attr_2s), dim=1)

    #####################################################
    # Randomize head-tail of edges at the batch level
    #####################################################

    if np.random.binomial(1, 0.5):
        batch = torch.cat((batch_data_1, batch_data_2), dim=1)
    else:
        batch = torch.cat((batch_data_2, batch_data_1), dim=1)

    return batch, cell_lines


def get_layer_dims(predictor_layers, fp_dim, attr_dim, prot_dim, with_fp=False, with_expr=False, with_prot=False):
    if with_expr:
        predictor_layers[0] += attr_dim * 2
    else:
        predictor_layers[0] += 2
    if with_fp:
        predictor_layers[0] += fp_dim * 2
    if with_prot:
        predictor_layers[0] += prot_dim * 2
    return predictor_layers