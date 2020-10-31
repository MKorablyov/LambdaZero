from torch_geometric.nn import SAGPooling, TopKPooling, GCNConv
from torch_sparse import SparseTensor

"""
This file contains some really naive implementations of a couple
pooling layers from pytorch_geometric (PyG).  The implementations are
effectively copy/pastes of the PyG implementations, but with a couple
lines different to use sparse tensors where memory explosion would happen
with dense tensors.  Eventually, when I have more time, this will be reviisited
and reimplemented to actually make use of spase tensors most efficiently.
But I'm short on time now, so this will have to do.
"""

def maybe_num_nodes(index: torch.Tensor,
                    num_nodes: Optional[int] = None) -> int:
    return int(index.max()) + 1 if num_nodes is None else num_nodes

def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = torch.nonzero(x > scores_min).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes, ),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm

def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes, ), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr

class SAGPooling(SAGPooling):
   def __init__(self, in_channels, pooling_ratio=0.5, GNN=GCNConv, min_score=None,
                multiplier=1, nonlinearity=torch.tanh, **kwargs):
        super().__init__(in_channels, ratio, GNN, min_score,
                         multiplier, nonlinearity, **kwargs)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        is_sparse = isinstance(edge_index, SparseTensor)
        if batch is None:
            if is_sparse:
                batch = torch.zeros(x.size(0), dtype=edge_index.dtype(), device=edge_index.device())
            else:
                batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        if is_sparse:
            row, col, edge_attr = edge_index.t().coo()
            edge_index = torch.stack((row, col), dim=0)

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        if is_sparse:
            edge_index = SparseTensor.from_edge_index(edge_index, edge_attr).t()
            edge_attr = None

        return x, edge_index, edge_attr, batch, perm, score[perm]

class TopKPooling(TopKPooling):
    def __init__(self, in_channels, pooling_ratio=0.5, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh):
        super().__init__(in_channels, pooling_ratio, min_score, multiplier, nonlinearity)

    def forward(self, x, edge_index, edge_attr=None, batch=None, attn=None):
        """"""
        is_sparse = isinstance(edge_index, SparseTensor)
        if batch is None:
            if is_sparse:
                batch = torch.zeros(x.size(0), dtype=edge_index.dtype(), device=edge_index.device())
            else:
                batch = edge_index.new_zeros(x.size(0))

        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = (attn * self.weight).sum(dim=-1)

        if self.min_score is None:
            score = self.nonlinearity(score / self.weight.norm(p=2, dim=-1))
        else:
            score = softmax(score, batch)

        perm = topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        if is_sparse:
            row, col, edge_attr = edge_index.t().coo()
            edge_index = torch.stack((row, col), dim=0)

        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        if is_sparse:
            edge_index = SparseTensor.from_edge_index(edge_index, edge_attr).t()
            edge_attr = None

        return x, edge_index, edge_attr, batch, perm, score[perm]

