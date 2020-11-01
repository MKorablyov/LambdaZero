import torch
from torch.nn import functional as F
import numpy as np


########################################################################################################################
# Modules
########################################################################################################################


class ResidualModule(torch.nn.Module):
    def __init__(self, ConvLayer, drug_channels, prot_channels,
                 pass_d2d_msg, pass_d2p_msg, pass_p2d_msg, pass_p2p_msg,
                 num_lyrs, data):

        super(ResidualModule, self).__init__()
        self.convs = torch.nn.ModuleList([
            ConvLayer(drug_channels, prot_channels, drug_channels,
                      prot_channels, pass_d2d_msg, pass_d2p_msg,
                      pass_p2d_msg, pass_p2p_msg, data)
            for _ in range(num_lyrs)
        ])

    def forward(self, h_drug, h_prot, data, drug_drug_batch):
        for i in range(len(self.convs)):
            out_drug, out_prot = self.convs[i](h_drug, h_prot, data, drug_drug_batch)

            if i == len(self.convs) - 1:
                out_drug, out_prot = out_drug + h_drug, out_prot + h_prot

            if i != len(self.convs) - 1:
                out_drug, out_prot = F.relu(out_drug), F.relu(out_prot)

        return out_drug, out_prot


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


class LRGAWithDimReduce(LowRankAttention):
    def __init__(self, out_dim, k, d, dropout):
        super().__init__(k, d, dropout)
        self.dim_reduce = torch.nn.Linear((2 * k) + out_dim, out_dim)

    def forward(self, pre_conv_x, post_conv_x):
        att = super().forward(pre_conv_x)
        return self.dim_reduce(torch.cat((att, post_conv_x)))


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

def build_attentions(data, config):
    kwargs = {
        'out_dim': config['residual_layers_dim'],
        'k': config['attention_rank'],
        'd': config['residual_layers_dim'],
        'dropout': config['dropout_proba'],
    }

    return [LRGAWithDimReduce(**kwargs) for _ in range(config['num_attention_modules'])]

def build_poolings(data, config):
    pooling_obj = config['pooling_layer']
    req_param_to_conf_param = {
        'GNN': 'pooling_gnn',
        'ratio': 'pooling_ratio',
    }

    return [
        construct_pooling(pooling_obj, config, data, req_param_to_conf_param)
        for _ in range(config['num_pooling_modules'])
    ]

def build_non_residual_convs(data, config, is_residual):
    kwargs = {
        'in_drug_channels': data.x_drugs.shape[1],
        'in_prot_channels': config['prot_emb_dim'],
        'out_drug_channels': config['residual_layers_dim'],
        'out_prot_channels': config['residual_layers_dim'],
        'pass_d2d_msg': config['pass_d2d_msg'],
        'pass_d2p_msg': config['pass_d2p_msg'],
        'pass_p2d_msg': config['pass_p2d_msg'],
        'pass_p2p_msg': config['pass_p2p_msg'],
        'data': data,
    }

    return [ProtDrugProtProtConvLayer(**kwargs) for _ in range(config['num_non_residual_convs'])]

def build_residual_modules(data, config):
    kwargs = {
        'ConvLayer': ProtDrugProtProtConvLayer,
        'drug_channels': config['residual_layers_dim'],
        'prot_channels': config['residual_layers_dim'],
        'pass_d2d_msg': config['pass_d2d_msg'],
        'pass_d2p_msg': config['pass_d2p_msg'],
        'pass_p2d_msg': config['pass_p2d_msg'],
        'pass_p2p_msg': config['pass_p2p_msg'],
        'num_lyrs': config['residual_block_length'],
        'data': data,
    }

    return [ResidualModule(**kwargs) for _ in range(config['num_residual_blocks'])]

