import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter, ParameterList
import numpy as np
from LambdaZero.examples.drug_comb.models.utils import get_batch, get_layer_dims


########################################################################################################################
# Abstract Predictor
########################################################################################################################


class AbstractPredictor(torch.nn.Module):
    def __init__(self, data, config):
        self.num_cell_lines = len(torch.unique(data.ddi_edge_classes))
        self.device = config["device"]
        self.with_fp = config["with_fp"]
        self.with_expr = config["with_expr"]
        self.with_prot = config["with_prot"]
        self.layer_dims = self.get_layer_dims(config["predictor_layers"], fp_dim=data.x_drugs.shape[1],
                                              attr_dim=data.ddi_edge_attr.shape[1] // 2,
                                              prot_dim=data.x_prots.shape[0])

        # Create dictionary linking drug to targets
        self.drug2target_dict = {i: [] for i in range(data.x_drugs.shape[0])}
        for edge in data.dpi_edge_idx.T:
            self.drug2target_dict[int(edge[0])].append(int(edge[1]) - data.x_drugs.shape[0])

        super(AbstractPredictor, self).__init__()

    def forward(self, data, drug_drug_batch, h_drug, h_prot, n_forward_passes=1):
        batch, cell_lines = self.get_batch(data, drug_drug_batch, h_drug, h_prot)

        out = torch.empty([drug_drug_batch[0].shape[0], 0]).to(self.device)
        # Perform several forward passes for MC dropout
        for i in range(n_forward_passes):
            x = self.single_forward_pass(batch, cell_lines)
            out = torch.cat((out, x), dim=1)

        return out

    def get_layer_dims(self, predictor_layers, data):
        return get_layer_dims(predictor_layers,
                              fp_dim=data.x-drugs.shape[1],
                              attr_dim=data.ddi_edge_attr.shape[1] // 2,
                              prot_dim=data.x_prots.shape[0],
                              with_fp=self.with_fp, i
                              with_expr=self.with_expr,
                              with_prot=self.with_prot)

    def get_batch(self, data, drug_drug_batch, h_drug, h_prot):
        return get_batch(data, drug_drug_batch, h_drug, self.drug2target_dict,
                         with_fp=self.with_fp, with_expr=self.with_expr, with_prot=self.with_prot)

    def single_forward_pass(self, batch, cell_lines):
        raise NotImplementedError


class MLPAbstractPredictor(AbstractPredictor):
    def __init__(self, data, config):
        super(MLPAbstractPredictor, self).__init__(data, config)
        assert self.layer_dims[-1] == 1

    def single_forward_pass(self, batch, cell_lines):
        raise NotImplementedError


########################################################################################################################
# Inner Product Predictor (does not implement dropout)
########################################################################################################################


class InnerProductPredictor(AbstractPredictor):
    def __init__(self, data, config):
        """
        Non linear transformation (linear layer + Relu) applied to each drug embedding independently
        Then the score is predited as <h_1 , G . h_2> where G is learnt. We enforce G symmetric

        All transformations are cell line specific
        """
        super(InnerProductPredictor, self).__init__(data, config)
        assert len(self.layer_dims) == 2
        self.predictor_lin = Parameter(1 / 100 * torch.randn((self.num_cell_lines, self.layer_dims[0],
                                                              self.layer_dims[1])))
        self.predictor_prod = Parameter(1 / 100 * torch.randn((self.num_cell_lines, self.layer_dims[1],
                                                               self.layer_dims[1])))  # Mat for outer product

    def get_batch(self, data, drug_drug_batch, h_drug, h_prot):
        batch_size = drug_drug_batch[0].shape[0]

        drug_1s = drug_drug_batch[0][:, 0]  # Edge-tail drugs in the batch
        drug_2s = drug_drug_batch[0][:, 1]  # Edge-head drugs in the batch
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        # Get drug embeddings
        if h_drug is not None:
            h_drug_1s = h_drug[drug_1s]  # Embeddings of tail drugs in the batch
            h_drug_2s = h_drug[drug_2s]  # Embeddings of head drugs in the batch
        else:  # The embedding of the drug is the fingerprint
            h_drug_1s = data.x_drugs[drug_1s]
            h_drug_2s = data.x_drugs[drug_2s]

        h_drug_1s = h_drug_1s.reshape(batch_size, 1, -1)  # Trick to allow broadcasting with matmul
        h_drug_2s = h_drug_2s.reshape(batch_size, 1, -1)  # Trick to allow broadcasting with matmul

        # Build G to be symmetric
        G = self.predictor_prod[cell_lines] + self.predictor_prod[cell_lines].permute(0, 2, 1)

        batch = (h_drug_1s, h_drug_2s, G)

        return batch, cell_lines

    def single_forward_pass(self, batch, cell_lines):
        h_drug_1s, h_drug_2s, G = batch

        # Cell line specific linear layer
        h_drug_1s = F.relu(h_drug_1s.matmul(self.predictor_lin[cell_lines]))
        h_drug_2s = F.relu(h_drug_2s.matmul(self.predictor_lin[cell_lines]))

        h_drug_2s = h_drug_2s.reshape(cell_lines.shape[0], -1, 1)  # Trick to allow broadcasting with matmul

        return h_drug_1s.matmul(G).matmul(h_drug_2s)[:, 0]

    def get_layer_dims(self, predictor_layers, data):
        predictor_layers[0] = predictor_layers[0] // 2

        return predictor_layers


########################################################################################################################
# Basic MLP Predictor
########################################################################################################################


class MLPPredictor(MLPAbstractPredictor):
    """
    Does not take into account cell line
    """
    def __init__(self, data, config):
        super(MLPPredictor, self).__init__(data, config)

        layers = []

        for i in range(len(self.layer_dims) - 1):
            layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i+1]))
            if i != len(self.layer_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(torch.nn.Dropout(p=config["dropout_proba"]))

        self.mlp = nn.Sequential(*layers)

    def single_forward_pass(self, batch, cell_lines):
        return self.mlp(batch)


########################################################################################################################
# Concentration Only MLP Predictor (does not implement dropout)
########################################################################################################################


class ConcentrationOnlyPredictor(MLPPredictor):
    def __init__(self, data, config):
        super(ConcentrationOnlyPredictor, self).__init__(data, config)
        # Make it clear that fingerprints and expressions are not used
        assert not self.with_fp and not self.with_expr, "'with_fp' and 'with_expr' should be set to " \
                                                        "False when using ConcentrationOnly"

    def get_layer_dims(self, predictor_layers, data):
        predictor_layers[0] = 2

        return predictor_layers

    def get_batch(self, data, drug_drug_batch, h_drug, h_prot):
        n_attr = drug_drug_batch[2].shape[1] // 2
        cell_lines = drug_drug_batch[1]  # Cell line of all examples in the batch

        conc_1 = drug_drug_batch[2][:, n_attr-1][:, None]
        conc_2 = drug_drug_batch[2][:, -1][:, None]

        if np.random.binomial(1, 0.5):  # Randomize head-tail of edges at the batch level
            batch = torch.cat((conc_1, conc_2), dim=1)
        else:
            batch = torch.cat((conc_2, conc_1), dim=1)

        return batch, cell_lines


class PPIPredictor(MLPPredictor):
    def __init__(self, data, config):
        super().__init__(data, config)

    def get_layer_dims(self, predictor_layers, data):
        predictor_layers[0] = data.h_prot.shape[0] * data.h_prot.shape[1]
        return predictor_layers

    def get_batch(self, data, drug_drug_batch, h_drug, h_prot):
        return h_prot.flatten()


########################################################################################################################
# Predictors with conditioning
########################################################################################################################


class FilmModule(torch.nn.Module):
    def __init__(self, num_cell_lines, out_dim):
        super(FilmModule, self).__init__()
        film_init = 1 / 100 * torch.randn(num_cell_lines, 2 * out_dim)
        film_init = film_init + torch.Tensor([([1] * out_dim) + ([0] * out_dim)])

        self.film = Parameter(film_init)

    def forward(self, x, cell_lines):
        return self.film[cell_lines][:, :x.shape[1]] * x + self.film[cell_lines][:, x.shape[1]:]


class FilmMLPPredictor(MLPAbstractPredictor):
    """
    Takes into account cell line with Film conditioning. Takes fingerprints as input as well
    """

    def __init__(self, data, config):
        super(FilmMLPPredictor, self).__init__(data, config)

        layers = []

        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i + 1]

            layers.append(nn.Linear(in_dim, out_dim))
            if i != len(self.layer_dims) - 2:
                layers.append(FilmModule(self.num_cell_lines, out_dim))
                layers.append(nn.ReLU())
                layers.append(torch.nn.Dropout(p=config["dropout_proba"]))

        self.layers = nn.ModuleList(layers)

    def single_forward_pass(self, batch, cell_lines):
        x = batch
        for i in range(len(self.layers)):
            if type(self.layers[i]) == FilmModule:
                x = self.layers[i](x, cell_lines)
            else:
                x = self.layers[i](x)

        return x


class SharedLayersMLPPredictor(MLPAbstractPredictor):
    """
    Takes into account cell line. All layers but the last two are shared between
    cell lines. The last two layers are cell line specific
    The forward passes are all computed in parallel
    Dropout only happens in shared layers
    """

    def __init__(self, data, config):
        super(SharedLayersMLPPredictor, self).__init__(data, config)

        last_layers = []
        shared_layers = []
        offsets = []

        # Shared layers
        for i in range(len(self.layer_dims) - 3):
            shared_layers.append(nn.Linear(self.layer_dims[i], self.layer_dims[i + 1]))
            shared_layers.append(nn.ReLU())
            shared_layers.append(torch.nn.Dropout(p=config["dropout_proba"]))

        # Cell line specific layers
        for i in range(len(self.layer_dims) - 3, len(self.layer_dims) - 1):
            last_layers.append(Parameter(1 / 100 * torch.randn((self.num_cell_lines, self.layer_dims[i + 1],
                                                                self.layer_dims[i]))))
            offsets.append(Parameter(1 / 100 * torch.randn((self.num_cell_lines, self.layer_dims[i + 1], 1))))

        self.shared_layers = nn.ModuleList(shared_layers)
        self.last_layers = ParameterList(last_layers)
        self.offsets = ParameterList(offsets)

    def single_forward_pass(self, batch, cell_lines):
        batch_size = batch.shape[0]
        x = batch
        # Forward in shared layers
        for i in range(len(self.shared_layers)):
            x = self.shared_layers[i](x)

        x = x.reshape(batch_size, -1, 1)  # Trick to allow broadcasting with matmul

        # Forward in cell_line specific MLPs in parallel
        for i in range(len(self.last_layers)):
            x = self.last_layers[i][cell_lines].matmul(x) + self.offsets[i][cell_lines]
            if i != len(self.last_layers) - 1:
                x = F.relu(x)

        return x[:, 0]
