import torch
from torch import nn
from torch_geometric.nn import MessagePassing, Set2Set, global_add_pool
from torch_geometric.typing import Adj, Size, OptTensor, Tensor
from LambdaZero.examples.lightrl.models.model_base import ModelBase
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


class EGNNetMhead(ModelBase):
    def __init__(self, cfg: Namespace, obs_shape, action_space):
        super().__init__(cfg, obs_shape, action_space)

        levels = getattr(cfg, "levels", 3)
        num_feat = getattr(cfg, "num_feat", 14)
        pos_dim = getattr(cfg, "pos_dim", 3)
        edge_attr_dim = getattr(cfg, "edge_attr_dim", 0)
        dim = getattr(cfg, "dim", 64)
        m_dim = getattr(cfg, "m_dim", 64)
        infer_edges = getattr(cfg, "infer_edges", False)
        settoset = getattr(cfg, "settoset", False)
        control_exp = getattr(cfg, "control_exp", False)
        num_out_per_stem = getattr(cfg, "num_out_per_stem", 105)
        num_out_per_mol = getattr(cfg, "num_out_per_mol", 2)

        feats_dim = num_feat
        self.n_layers = n_layers = levels
        self.egnn_layers = nn.ModuleList()
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.edge_attr_dim = edge_attr_dim
        self.m_dim = m_dim
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

        self.node2stem = nn.ModuleList([
            nn.Sequential(nn.Linear(feats_dim, feats_dim), _act(), nn.Linear(feats_dim, num_out_per_stem))
            for _ in range(50)])
        self.node2jbond = nn.ModuleList([
            nn.Sequential(nn.Linear(feats_dim, feats_dim), _act(), nn.Linear(feats_dim, 1))
            for _ in range(50)])

    def forward(self, inputs, rnn_hxs, masks, vec_data=None):
        data = inputs.mol_graph

        out = data.x  # or another activation here
        if self.do_feat_proj:
            out = self.feat_proj(out)

        for il, layer in enumerate(self.egnn_layers):
            out = layer(out, data.pos, data.edge_index, data.edge_attr, size=data.batch)

        data.stem_preds = self.node2stem(out[data.stem_atmidx])
        data.jbond_preds = self.node2jbond(out[data.jbond_atmidx.flatten()])\
            .reshape((data.jbond_atmidx.shape)).mean(1)  # mean pooling of the 2 jbond atom preds

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

        stop_logit = out[:, 1:2]
        break_logits = data.jbond_preds.reshape((data.num_graphs, -1))
        add_logits = data.stem_preds.reshape((data.num_graphs, -1))

        actor_logits = torch.cat([stop_logit, break_logits, add_logits], 1)

        value = out[:, :1]

        return value, actor_logits, rnn_hxs


