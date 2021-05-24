import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree
from torch_scatter import scatter


# adapted from https://github.com/rusty1s/pytorch_geometric/blob/10c486a642fe47d4a123c9b8476c1f9282ad849d/torch_geometric/nn/conv/pna_conv.py
class PNA:
    """ Principal Neighbourhood Aggregation: https://arxiv.org/pdf/2004.05718.pdf"""
    def __init__(self, avg_deg_lin, avg_deg_log,
                 aggregators=None,
                 scalers=None):
        self.avg_deg_lin = avg_deg_lin
        self.avg_deg_log = avg_deg_log
        self.aggregators = ('mean', 'min', 'max', 'std') if aggregators is None else aggregators
        self.scalers = ('identity', 'amplification', 'attenuation') if scalers is None else scalers

        # self.node_dim - passed from MessagePassing init.

    @staticmethod
    def get_avg_degree(flow, deg=None, graphs=None):
        assert flow in ['source_to_target', 'target_to_source'], "flow must be either 'source_to_target' or 'target_to_source'"
        assert deg is not None or graphs is not None, "Either deg or graphs should be supplied"
        idx = 1 if flow == 'source_to_target' else 0
        deg = torch.cat([degree(graph.edge_index[idx], num_nodes=graph.num_nodes) for graph in graphs]) if deg is None else deg
        avg_deg_lin = deg.mean().item()
        avg_deg_log = (deg + 1).log().mean().item()
        return {'avg_deg_lin': avg_deg_lin, 'avg_deg_log': avg_deg_log}

    # match torch_geometric interface
    def aggregate(self, inputs, index, dim_size):
        origin_node_degrees = degree(index, num_nodes=dim_size)
        origin_node_degrees = origin_node_degrees.clamp_(1).view(-1, 1)

        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(torch.relu(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')
            outs.append(out)
        out = torch.cat(outs, dim=-1)

        outs = []
        for scaler in self.scalers:
            if scaler == 'identity':
                pass
            elif scaler == 'amplification':
                out = out * (torch.log(origin_node_degrees + 1) / self.avg_deg_log)
            elif scaler == 'attenuation':
                out = out * (self.avg_deg_log / torch.log(origin_node_degrees + 1))
            elif scaler == 'linear':
                out = out * (origin_node_degrees / self.avg_deg_lin)
            elif scaler == 'inverse_linear':
                out = out * (self.avg_deg_lin / origin_node_degrees)
            else:
                raise ValueError(f'Unknown scaler "{scaler}".')
            outs.append(out)

        return torch.cat(outs, dim=-1)


# original pytorch_geometric implementation has preset network model
class PNAConv(PNA, MessagePassing):
    def __init__(self, avg_deg_lin, avg_deg_log, in_channels, msg_channels, out_channels,
                 nn, root_weight=True, bias=True, aggregators=None, scalers=None, flow='source_to_target'):
        MessagePassing.__init__(self, flow=flow)
        PNA.__init__(self, avg_deg_lin, avg_deg_log, aggregators=aggregators, scalers=scalers)

        self.in_channels = in_channels
        self.msg_channels = msg_channels
        self.out_channels = out_channels
        self.nn = nn

        factor = len(self.aggregators) * len(self.scalers)
        self.lin_aggr = torch.nn.Linear(factor * msg_channels, out_channels, bias=bias)

        if root_weight:
            self.root = torch.nn.Linear(in_channels, out_channels, bias=False)  # avoid double bias
        else:
            self.root = None

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.root:
            out += self.root(x)
        return out

    def message(self, x_j, edge_attr):
        weight = self.nn(edge_attr).view(-1, self.in_channels, self.msg_channels)  # [n_msg, in_ch * msg_ch] -> [n_msg, in_ch, msg_ch]
        return torch.einsum('ni,nim->nm', x_j, weight)

    def aggregate(self, inputs, index, dim_size):
        out = PNA.aggregate(self, inputs, index, dim_size)
        return self.lin_aggr(out)
