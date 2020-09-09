import torch
from torch.nn import functional as F
from torch.nn import Parameter
from recover.models.utils import ResidualModule, LowRankAttention


########################################################################################################################
# Giant Graph GCN
########################################################################################################################


class GiantGraphGCN(torch.nn.Module):
    def __init__(self, data, config):
        super(GiantGraphGCN, self).__init__()

        self.device = config["device"]

        # Learnable protein embeddings
        self.prot_emb = Parameter(1 / 100 * torch.randn((data.x_prots.shape[0], config["prot_emb_dim"])))

        # First Graph convolution layer
        self.conv1 = config["conv_layer"](data.x_drugs.shape[1], config["prot_emb_dim"],
                                          config["residual_layers_dim"], config["residual_layers_dim"], data)

        # Low rank attention
        self.has_attention = config["attention"]
        if self.has_attention:
            self.low_rank_attention = []
            self.low_rank_attention.append(LowRankAttention(k=config["attention_rank"],
                                                            d=config["residual_layers_dim"], dropout=0))

        # Residual layers
        self.residual_layers = []
        drug_channels = config["residual_layers_dim"]
        for i in range(config['num_res_layers']):
            if self.has_attention:
                drug_channels += 2 * config["attention_rank"]
                self.low_rank_attention.append(LowRankAttention(k=config["attention_rank"],
                                                                d=drug_channels, dropout=0))
            self.residual_layers.append(ResidualModule(ConvLayer=config["conv_layer"],
                                                       drug_channels=drug_channels,
                                                       prot_channels=config["residual_layers_dim"],
                                                       data=data))

        self.residual_layers = torch.nn.ModuleList(self.residual_layers)
        if self.has_attention:
            self.low_rank_attention = torch.nn.ModuleList(self.low_rank_attention)

        # Response Predictor on top of GCN
        predictor_layers = [drug_channels * 2 + config["attention_rank"] * 4 *
                            self.has_attention + data.ddi_edge_attr.shape[1]] + config["predictor_layers"]

        self.predictor = config["predictor"](data=data, num_cell_lines=len(torch.unique(data.ddi_edge_classes)),
                                             layer_dims=predictor_layers)
        self.criterion = torch.nn.MSELoss()

    def forward(self, data, drug_drug_batch):
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
                att = self.low_rank_attention[i+1](h_drug[:data.is_drug.sum()])
                att = torch.cat((att, torch.zeros(data.x_prots.shape[0], att.shape[1]).to(self.device)))
                h_drug_next = torch.cat((h_drug_next, att), dim=1)
            h_drug, h_prot = F.relu(h_drug_next), F.relu(h_prot_next)

        # Predict scores
        return self.predictor(data, drug_drug_batch, h_drug)

    def loss(self, output, drug_drug_batch):
        ground_truth_scores = drug_drug_batch[3]

        return self.criterion(output, ground_truth_scores)
