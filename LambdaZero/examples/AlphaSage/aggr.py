from typing import Optional, List, Union
import torch
from torch.nn import Sequential, Linear, ReLU, Dropout
from torch.nn import BatchNorm1d, LayerNorm, InstanceNorm1d

class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super(MLP, self).__init__(*m)

class GRUAggr(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, gru_layers):
        super(GRUAggr, self).__init__()
        self.gru = torch.nn.GRU(in_dim, hidden_size, num_layers=gru_layers, batch_first=True, bidirectional=True)
        self.lin = torch.nn.Linear(hidden_size * 2, out_dim)

    def forward(self, inputs):
        out, _ = self.gru(inputs)
        out = self.lin(out)[:,0,:]
        return out



#gru = GRUAggr(3, 7, 128, 2)
#inputs = torch.randn(10,1,3)
#print(gru(inputs).shape)