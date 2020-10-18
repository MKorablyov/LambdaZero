import torch
import numpy as np
from torch.nn import functional as F
from torch.nn import Parameter
from LambdaZero.examples.drug_comb.models.utils import ResidualModule, LowRankAttention
from LambdaZero.examples.drug_comb.utils import get_dropout_modules_recursive


########################################################################################################################
# Abstract Model
########################################################################################################################

class AbstractModel(torch.nn.Module):
    def __init__(self, data, config):
        self.device = config["device"]
        super(AbstractModel, self).__init__()
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        raise NotImplementedError

    def enable_dropout(self):
        for m in get_dropout_modules_recursive(self):
            m.train()

    def loss(self, output, drug_drug_batch):
        ground_truth_scores = drug_drug_batch[3][:, None]
        ground_truth_scores = torch.cat([ground_truth_scores]*output.shape[1], dim=1)

        return self.criterion(output, ground_truth_scores)

    # Virtual methods to be overridden by implementations of this
    # class if need be
    def enable_periodic_backprop(self):
        pass

    def disable_periodic_backprop(self):
        pass


########################################################################################################################
# Baselines with no GCN
########################################################################################################################


class Dummy(AbstractModel):
    def __init__(self, data, config):
        super(Dummy, self).__init__(data, config)
        self.param = torch.nn.Linear(1, 1)  # So that optimizer does not get an empty parameter list

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        batch_size = drug_drug_batch[0].shape[0]

        return torch.zeros(batch_size, requires_grad=True) + data.ddi_edge_attr[:, 0].mean()

    def loss(self, output, drug_drug_batch):
        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output.to(self.device), ground_truth_scores)


class Baseline(AbstractModel):
    def __init__(self, data, config):
        super(Baseline, self).__init__(data, config)
        assert not config["with_fp"], "When using Baseline, 'with_fp' should be set to False"

        # Compute dimension of input for predictor
        config["predictor_layers"] = [2 * data.x_drugs.shape[1]] + \
                                     config["predictor_layers"]

        self.predictor = config["predictor"](data, config)

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        return self.predictor(data, drug_drug_batch, h_drug=None, n_forward_passes=n_forward_passes)


########################################################################################################################
# Base Periodic Backprop Model
########################################################################################################################

class BasePeriodicBackpropModel(AbstractModel):
    def __init__(self, data, config):
        super(BasePeriodicBackpropModel, self).__init__(data, config)

        self.do_periodic_backprop = config["do_periodic_backprop"] if "do_periodic_backprop" in config else False
        self.backprop_period = config["backprop_period"] if "backprop_period" in config else None
        self.periodic_backprop_enabled = False
        self.curr_backprop_status = False
        self.backprop_iter = 0

    def forward(self, data, drug_drug_batch, n_forward_passes=1):
        if self.periodic_backprop_enabled:
            should_enable = self.backprop_iter % self.backprop_period == 0
            self.set_backprop_enabled_status(should_enable)
            self.backprop_iter += 1

        return self._forward(data, drug_drug_batch, n_forward_passes)

    def _forward(self, data, drug_drug_batch, n_forward_passes=1):
        raise NotImplementedError

    def set_backprop_enabled_status(self, status):
        if status != self.curr_backprop_status:
            for var in self.get_periodic_backprop_vars():
                var.requires_grad = status

            self.curr_backprop_status = status

    def get_periodic_backprop_vars(self):
        raise NotImplementedError

    def enable_periodic_backprop(self):
        assert self.backprop_period is not None
        self.periodic_backprop_enabled = self.do_periodic_backprop

    def disable_periodic_backprop(self):
        assert self.backprop_period is not None
        self.periodic_backprop_enabled = False

########################################################################################################################
# Giant Graph GCN
########################################################################################################################


class GiantGraphGCN(BasePeriodicBackpropModel):
    def __init__(self, data, config):
        super(GiantGraphGCN, self).__init__(data, config)

        # Learnable protein embeddings
        self.prot_emb = Parameter(1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"])))

        # First Graph convolution layer
        self.conv1 = config["conv_layer"](data.x_drugs.shape[1], config["prot_emb_dim"],
                                          config["residual_layers_dim"], config["residual_layers_dim"],
                                          config['pass_d2d_msg'], config['pass_d2p_msg'], config['pass_p2d_msg'],
                                          config['pass_p2p_msg'], data)

        # First Low rank attention layer
        self.has_attention = config["attention"]
        if self.has_attention:
            self.low_rank_attention = []
            self.low_rank_attention.append(LowRankAttention(k=config["attention_rank"],
                                                            d=data.x_drugs.shape[1],
                                                            dropout=config["dropout_proba"]))

        # Residual layers
        self.residual_layers = []
        drug_channels = config["residual_layers_dim"]
        for i in range(config['num_res_layers']):
            if self.has_attention:
                # If attention is used, we must increase the number of drug channels
                drug_channels += 2 * config["attention_rank"]
                self.low_rank_attention.append(LowRankAttention(k=config["attention_rank"],
                                                                d=drug_channels, dropout=config["dropout_proba"]))

            self.residual_layers.append(ResidualModule(ConvLayer=config["conv_layer"],
                                                       drug_channels=drug_channels,
                                                       prot_channels=config["residual_layers_dim"],
                                                       pass_d2d_msg=config["pass_d2d_msg"],
                                                       pass_d2p_msg=config["pass_d2p_msg"],
                                                       pass_p2d_msg=config["pass_p2d_msg"],
                                                       pass_p2p_msg=config["pass_p2p_msg"],
                                                       data=data))

        # convert to ModuleList
        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_attention:
            self.low_rank_attention = torch.nn.ModuleList(self.low_rank_attention)

        # Compute dimension of input for predictor
        config["predictor_layers"] = [drug_channels * 2 + config["attention_rank"] * 4 *
                                      self.has_attention] + config["predictor_layers"]

        # Response Predictor on top of GCN
        self.predictor = config["predictor"](data, config)

    def get_periodic_backprop_vars(self):
        yield self.prot_emb
        yield from self.conv1.parameters()
        yield from self.residual_layers.parameters()
        if self.has_attention:
            yield from self.low_rank_attention.parameters()

    def _forward(self, data, drug_drug_batch, n_forward_passes=1):
        # For compatibility with Message passing, h_drugs and h_prot need to have the same 1st dimension
        h_drug = torch.cat((data.x_drugs, torch.zeros(data.x_prots.shape[0], data.x_drugs.shape[1]).to(self.device)))
        h_prot = torch.cat((torch.zeros(data.x_drugs.shape[0], self.prot_emb.shape[1]).to(self.device), self.prot_emb))

        ##########################################
        # GNN forward pass
        ##########################################
        h_drug_next, h_prot_next = self.conv1(h_drug, h_prot, data)
        if self.has_attention:
            att = self.low_rank_attention[0](h_drug[:data.is_drug.sum()])
            att = torch.cat((att, torch.zeros(data.x_prots.shape[0], att.shape[1]).to(self.device)))
            h_drug_next = torch.cat((h_drug_next, att), dim=1)
        h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        for i in range(len(self.residual_layers)):
            h_drug_next, h_prot_next = self.residual_layers[i](h_drug, h_prot, data)
            if self.has_attention:
                att = self.low_rank_attention[i + 1](h_drug[:data.is_drug.sum()])
                att = torch.cat((att, torch.zeros(data.x_prots.shape[0], att.shape[1]).to(self.device)))
                h_drug_next = torch.cat((h_drug_next, att), dim=1)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # Predict scores
        return self.predictor(data, drug_drug_batch, h_drug, n_forward_passes=n_forward_passes)
