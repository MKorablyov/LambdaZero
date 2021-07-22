import torch
import torch.nn as nn
from argparse import Namespace


class GFlowModelBase(nn.Module):
    def __init__(self, cfg: Namespace, **kwargs):
        super(GFlowModelBase, self).__init__()

