import time
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_max, scatter_sum
from mlp import MLP

# src = torch.tensor([[17, 19,]
#                     [7, 11, 13],
#                      [3, 5, 23, 29, 31]
#                      ])
# index = torch.tensor([[0,0,0,1,1]])
#
# segsum = scatter_sum(src, index, dim=-1)
# print(segsum)
# # segmax, argmax = scatter_max(src, index, dim=1)
# # print(segmax, argmax)
# time.sleep(1000)




class SimpleGCN(torch.nn.Module):
    def __init__(self, channels):
        super(SimpleGCN, self).__init__()
        self.mlp = MLP(channels)

    def message(self, x, adj): # [v,d] -> [e,d]
        edge_x = x[adj[0]]
        return edge_x

    def aggregate(self, edge_x, adj, num_nodes): # [e,d] -> [v,d]
        x = scatter(edge_x, adj[1], reduce="max", dim=0, dim_size=num_nodes)
        return x

    def combine(self, x1, x2):
        return self.mlp(x1 + x2)

    def forward(self, x1, adj):
        num_nodes = x1.shape[0]
        edge_x = self.message(x1, adj)
        x2 = self.aggregate(edge_x, adj, num_nodes)
        x3 = self.combine(x1,x2)
        return x3




if __name__ == "__main__":
    x1 = torch.arange(12,dtype=torch.float).reshape(4,3)

    edge_index = torch.tensor([[0, 1, 2, 3, 3],
                               [0, 0, 1, 1, 1]])

    conv = SimpleGCN([3, 7])
    conv(x1,edge_index)


# def test_sage_conv():
#     conv = SAGEConv(3,3)
#     conv.__fuse__ = False
#     out = conv(x1, edge_index)
#
#     print(out)
# test_sage_conv()