import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree, add_remaining_self_loops
import time

class RelationAwareLinear(torch.nn.Module):
    def __init__(self, num_relations, in_channels, out_channels, bias = True):
        self.w = Parameter(torch.Tensor(num_relatons, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(num_relations, out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, relations):
        out = x.mm(self.w[relations])
        if self.bias:
            out += self.bias[relations]

        return out

