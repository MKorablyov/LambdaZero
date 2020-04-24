from torch import nn
from torch.nn import Sequential, Linear, ReLU, GRU, functional as F
from torch_geometric.nn import NNConv, Set2Set


class MessagePassingNet(nn.Module):
    def __init__(self, num_feat=14, dim=64):
        super(MessagePassingNet, self).__init__()

        self.lin0 = nn.Linear(num_feat, dim)

        mlp = Sequential(Linear(4, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, mlp, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = nn.Linear(2 * dim, dim)
        self.lin2 = nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)