import torch
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch_geometric.utils import degree, add_remaining_self_loops
import time
import math

class RelationAwareLinear(torch.nn.Module):
    def __init__(self, num_relations, in_channels, out_channels, bias = True):
        super().__init__()
        self.w = Parameter(torch.Tensor(num_relations, in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(num_relations, out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.w, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.w)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, relations):
        # idx array for trick to do 3d tensor mult fast
        idxs = torch.arange(x.shape[0])

        out = x.matmul(self.w[relations])[idxs, 0, idxs, :]
        if self.bias is not None:
            out += self.bias[relations][:, 0, :]

        return out

