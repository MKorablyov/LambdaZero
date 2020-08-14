import torch
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_remaining_self_loops
import time

def joint_normalize(U, V_T):
    tmp_ones = torch.ones((V_T.shape[1], 1)).to(U.device)
    norm_factor = torch.mm(U, torch.mm(V_T, tmp_ones))
    norm_factor = (torch.sum(norm_factor) / U.shape[0]) + 1e-6
    return 1 / norm_factor

class LowRankAttention(torch.nn.Module):
    def __init__(self, in_channels, rank):
        super().__init__()
        self.u = torch.nn.Linear(in_channels, rank)
        self.v = torch.nn.Linear(in_channels, rank)
        self.z = torch.nn.Linear(in_channels, rank)
        self.t = torch.nn.Linear(in_channels, rank)

    def forward(self, x):
        """Do forward pass according to LRGA formula.

        We follow the equation
        [\frac{1}{\eta(x)} U(x) (V(x)^T Z(x)), T(x)]

        where
        \eta(x) = \frac{1}{n} (1^T U(x))(V(x)^T 1)

        and [x, y] represents the concatenation of x and y.

        Arguments
        ---------

        """
        U = self.u(x)
        V = self.v(x)
        Z = self.z(x)
        T = self.t(x)

        V_T = V.t()
        D = joint_normalize(U, V_T) # Normalization term

        pre_res = D * torch.mm(U, torch.mm(V_T, Z))
        return torch.cat((pre_res, T), dim=1)

