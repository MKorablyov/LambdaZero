import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, Set2Set, global_add_pool
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
from argparse import Namespace

_act = nn.SiLU
# _act = nn.LeakyReLU
# _act = nn.ELU
# _act = nn.ReLU
# _act = nn.SELU


class EGNNConv(MessagePassing):

    def __init__(self, feats_dim, pos_dim=3, edge_attr_dim=0, m_dim=64, infer_edges=False, control_exp=False,
                 aggr: str = 'add'):
        super().__init__(aggr=aggr)
        self.aggr = aggr
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.infer_edges = infer_edges
        self.control_exp = control_exp

        edge_input_dim = (feats_dim * 2) # + edge_attr_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, m_dim), # nn.Linear(edge_input_dim, edge_input_dim * 2),
            _act(),
            nn.Linear(m_dim, m_dim), # nn.Linear(edge_input_dim * 2, m_dim),
            _act()
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(feats_dim + m_dim, m_dim), # nn.Linear(feats_dim + m_dim, feats_dim * 2), # remove +m_dim for no-message passing
            _act(),
            nn.Linear(m_dim, feats_dim) # nn.Linear(feats_dim * 2, feats_dim)
        )

        if self.infer_edges:
            self.infer_edge_mlp = nn.Sequential(
                nn.Linear(m_dim, 1),
                nn.Sigmoid()
            )

        self.apply(self.init_)

        # self.pos_mlp = nn.Sequential( # for if we want to update positions
        #     nn.Linear(m_dim, m_dim * 4),
        #     dropout,
        #     nn.SiLU(),
        #     nn.Linear(m_dim * 4, 1)
        # )

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std=1e-3)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        hidden_out = self.propagate(edge_index, x=x)
        return hidden_out

    def message(self, x_i, x_j) -> Tensor:
        # if self.control_exp:
        #     # distance set to 0 in control experiment, or randlike, could also set x_i x_j to 0
        #     edge_attr.fill_(0)
        #     # edge_attr = torch.randn_like(edge_attr)
        #     # x_i.fill_(0)
        #     # x_j.fill_(0)
        # could try to embedding of edge_attr before torch.cat
        m_ij = self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
        # coor_w = self.pos_mlp(m_ij)
        if self.infer_edges:
            e_ij = self.infer_edge_mlp(m_ij)
            m_ij = e_ij * m_ij
        return m_ij  #, coor_w

    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        # m_ij, coor_wij = self.message(**msg_kwargs)
        m_ij = self.message(**msg_kwargs)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        if self.infer_edges:
            m_i = torch.zeros(int(edge_index.max()) + 1, dtype=m_ij.dtype, device=m_ij.device)
            m_i.scatter_add(self.node_dim, edge_index, m_ij)  # fixme to reconstruct edge_index for complete graph
        else:  # normal aggregation
            m_i = self.aggregate(m_ij, **aggr_kwargs)

        # coor_wi = self.aggregate(coor_wij, **aggr_kwargs)
        # coor_ri = self.aggregate(kwargs["rel_pos"], **aggr_kwargs)
        # node, pos = kwargs["x"], kwargs["pos"]
        # pos_out = pos + (coor_wi * coor_ri)
        # hidden_out = self.node_mlp(torch.cat([node, m_i], dim=-1))
        # hidden_out = hidden_out + node

        out = self.node_mlp(torch.cat([kwargs["x"], m_i], dim=-1)) + kwargs["x"]
        # out = self.node_mlp(kwargs["x"]) + kwargs["x"] # if we don't do message passing
        update_kwargs = self.inspector.distribute('update', coll_dict)
        return self.update(out, **update_kwargs)  # (out, pos_out)


class EGNNet(nn.Module):

    def __init__(self, n_layers=3, feats_dim=14, pos_dim=3, edge_attr_dim=0, m_dim=16,
                 infer_edges=False, settoset=False, control_exp=False):
        super().__init__()

        self.n_layers = n_layers
        self.egnn_layers = nn.ModuleList()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.m_dim = m_dim
        self.settoset = settoset

        for i in range(n_layers):
            EGCLayer = EGNNConv(feats_dim=feats_dim,
                                pos_dim=pos_dim,
                                edge_attr_dim=edge_attr_dim,
                                m_dim=m_dim, infer_edges=infer_edges, control_exp=control_exp)
            self.egnn_layers.append(EGCLayer)

        if self.settoset:
            self.set2set = Set2Set(feats_dim, processing_steps=3)
            self.lin1 = nn.Linear(2 * feats_dim, feats_dim)
            self.lin2 = nn.Linear(feats_dim, 1)
        else:
            self.lin1 = nn.Linear(feats_dim, m_dim)
            self.lin2 = nn.Linear(m_dim, feats_dim)
            self.lin3 = nn.Linear(feats_dim, m_dim)
            self.lin4 = nn.Linear(m_dim, 1)

    def forward(self, data):
        out = data.x  # or another activation here
        for layer in self.egnn_layers:
            out = layer(out, data.pos, data.edge_index, data.edge_attr, size=data.batch)

        if self.settoset:
            out = self.set2set(out, data.batch)
            out = nn.functional.silu(self.lin1(out))
            out = self.lin2(out)
        else:
            out = nn.functional.silu(self.lin1(out))
            out = self.lin2(out)
            out = global_add_pool(out, data.batch)
            out = nn.functional.silu(self.lin3(out))
            out = self.lin4(out)

        return out.view(-1)


class EGNNetRLBO(nn.Module):
    atom_feature_extractor = False

    def __init__(self, cfg: Namespace):
        super(EGNNetRLBO, self).__init__()
        feats_dim = num_feat = getattr(cfg, "num_feat", 14)
        dim = getattr(cfg, "dim", 64)
        m_dim = getattr(cfg, "m_dim", 64)
        self.num_out = num_out_per_mol = getattr(cfg, "num_out", 1)

        infer_edges = False
        settoset = False
        control_exp = False,

        self.n_layers = n_layers = getattr(cfg, "levels", 3)
        self.egnn_layers = nn.ModuleList()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim = getattr(cfg, "pos_dim", 3)
        self.edge_attr_dim = edge_attr_dim = getattr(cfg, "edge_attr_dim", 0)
        self.m_dim = getattr(cfg, "m_dim", 64)
        self.settoset = settoset

        self.do_feat_proj = False
        if dim > feats_dim:
            # First feature projection
            self.feat_proj = nn.Sequential(nn.Linear(feats_dim, dim))
            feats_dim = dim
            self.do_feat_proj = True

        for i in range(n_layers):
            EGCLayer = EGNNConv(feats_dim=feats_dim,
                                pos_dim=pos_dim,
                                edge_attr_dim=edge_attr_dim,
                                m_dim=m_dim, infer_edges=infer_edges, control_exp=control_exp)
            self.egnn_layers.append(EGCLayer)

        if self.settoset:
            self.set2set = Set2Set(feats_dim, processing_steps=3)
            self.lin1 = nn.Linear(2 * feats_dim, feats_dim)
            self.lin2 = nn.Linear(feats_dim, num_out_per_mol)
        else:
            self.lin1 = nn.Linear(feats_dim, m_dim)
            self.lin2 = nn.Linear(m_dim, feats_dim)
            self.lin3 = nn.Linear(feats_dim, m_dim)
            self.lin4 = nn.Linear(m_dim, num_out_per_mol)

        self._per_atom_out_size = feats_dim

    def forward(self, data):
        out = data.x  # or another activation here
        if self.do_feat_proj:
            out = self.feat_proj(out)

        for il, layer in enumerate(self.egnn_layers):
            out = layer(out, data.pos, data.edge_index, data.edge_attr, size=data.batch)

        per_atom_out = out

        if self.settoset:
            out = self.set2set(out, data.batch)
            out = _act()(self.lin1(out))
            out = self.lin2(out)
        else:
            out = _act()(self.lin1(out))
            out = self.lin2(out)
            out = global_add_pool(out, data.batch)
            out = _act()(self.lin3(out))
            out = self.lin4(out)

        if self.num_out == 1:
            out = out.view(-1)

        return out, per_atom_out

    @property
    def per_atom_out_size(self):
        return self._per_atom_out_size


class EGNNetDrop(nn.Module):

    def __init__(self, drop_last, drop_data, drop_weights, drop_prob, n_layers=3, feats_dim=14, pos_dim=3, edge_attr_dim=0, m_dim=16,
                 infer_edges=False, settoset=False, control_exp=False):
        super().__init__()
        self.drop_last = drop_last
        self.drop_data = drop_data
        self.drop_weights = drop_weights
        self.drop_prob = drop_prob

        self.n_layers = n_layers
        self.egnn_layers = nn.ModuleList()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.m_dim = m_dim
        self.settoset = settoset

        for i in range(n_layers):
            EGCLayer = EGNNConv(feats_dim=feats_dim,
                                pos_dim=pos_dim,
                                edge_attr_dim=edge_attr_dim,
                                m_dim=m_dim, infer_edges=infer_edges, control_exp=control_exp)
            self.egnn_layers.append(EGCLayer)

        if self.settoset:
            self.set2set = Set2Set(feats_dim, processing_steps=3)
            self.lin1 = nn.Linear(2 * feats_dim, feats_dim)
            self.lin2 = nn.Linear(feats_dim, 1)
        else:
            self.lin1 = nn.Linear(feats_dim, m_dim)
            self.lin2 = nn.Linear(m_dim, feats_dim)
            self.lin3 = nn.Linear(feats_dim, m_dim)
            self.lin4 = nn.Linear(m_dim, 1)

    def forward(self, data, do_dropout=True):
        if self.drop_data: data.x = F.dropout(data.x, training=do_dropout, p=self.drop_prob)
        out = data.x  # or another activation here
        # out.fill_(0)
        # data.pos.fill_(0)
        for layer in self.egnn_layers:
            out = layer(out, data.pos, data.edge_index, data.edge_attr, size=data.batch)
            if self.drop_weights: out = F.dropout(out, training=do_dropout, p=self.drop_prob)

        if self.settoset:
            out = self.set2set(out, data.batch)
            if self.drop_weights: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
            out = nn.functional.silu(self.lin1(out))
            if self.drop_last: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
            out = self.lin2(out)
        else:
            out = nn.functional.silu(self.lin1(out))
            out = self.lin2(out)
            if self.drop_weights: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
            out = global_add_pool(out, data.batch)
            if self.drop_weights: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
            out = nn.functional.silu(self.lin3(out))
            if self.drop_last: out = F.dropout(out, training=do_dropout, p=self.drop_prob)
            out = self.lin4(out)
        # return out.view(-1)
        return out
