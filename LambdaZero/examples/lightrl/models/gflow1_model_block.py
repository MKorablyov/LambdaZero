import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from argparse import Namespace
from LambdaZero.examples.lightrl.models.gflow_model_base import GFlowModelBase
from LambdaZero.examples.lightrl.models.model_base import ModelBase


def one_hot(idx, device, max_cnt):
    hot = torch.zeros(len(idx), max_cnt, device=device).scatter_(1, idx.flatten().unsqueeze(1), 1.)
    return hot


class GraphAgent(ModelBase):
    def __init__(self, cfg: Namespace, obs_shape, action_space, **kwargs):
        # super(GraphAgent, self).__init__(cfg, **kwargs)
        super().__init__(cfg, obs_shape, action_space)

        nemb = getattr(cfg, "nemb", 256)
        nvec = getattr(cfg, "nvec", 0)
        out_per_stem = getattr(cfg, "num_out_per_stem", 105)
        self.out_per_mol = out_per_mol = getattr(cfg, "num_out_per_mol", 2)
        num_conv_steps = getattr(cfg, "num_conv_steps", 10)
        version = getattr(cfg, "version", "v4")
        repr_type = getattr(cfg, "repr_type", "block_graph")
        self.max_steps = getattr(cfg, "max_steps", 10)
        self.zero_cond = getattr(cfg, "zero_cond", False)

        mdp_cfg = kwargs["mdp"]
        env = kwargs["env"]
        assert repr_type == "block_graph", "GraphAgent works on block_graph input"
        self._max_branches = env.max_branches
        self._num_blocks = env.num_blocks
        self._max_blocks = env.max_blocks

        print(version)
        if version == 'v5': version = 'v4'
        self.version = version
        self.embeddings = nn.ModuleList([
            nn.Embedding(mdp_cfg.num_true_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)])
        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')
        nvec_1 = nvec * (version == 'v1' or version == 'v3')
        nvec_2 = nvec * (version == 'v2' or version == 'v3')
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb))
        self.gru = nn.GRU(nemb, nemb)

        h_size = self.max_steps + 1

        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2 + h_size, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))
        self.bond2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2 + h_size, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, 1))

        self.global2pred = nn.Sequential(nn.Linear(nemb + h_size, nemb), nn.LeakyReLU(),
                                         nn.Linear(nemb, out_per_mol))
        if out_per_mol == 2:
            self.global2pred2 = nn.Sequential(nn.Linear(nemb + h_size, nemb), nn.LeakyReLU(),
                                              nn.Linear(nemb, out_per_mol))
        elif out_per_mol != 1:
            raise NotImplemented

        #self.set2set = Set2Set(nemb, processing_steps=3)
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'softmax'
        self.escort_p = 6

    def run_model(self, inputs, vec_data=None, do_stems=True, do_bonds=True, **kwargs):
        graph_data = inputs.mol_graph

        blockemb, stememb, bondemb = self.embeddings

        graph_data.x = blockemb(graph_data.x)
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes)

        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))
        out = graph_data.x

        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        elif self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out)

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # Aux info
        rsteps = one_hot(inputs.r_steps[graph_data.batch], inputs.r_steps.device, self.max_steps)
        rcond = inputs.rcond.float()[graph_data.batch].unsqueeze(1)
        if self.zero_cond:
            rsteps.zero_()
            rcond.zero_()
            print("Nope")
        out = torch.cat([out, rsteps, rcond], dim=1)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            stem_block_batch_idx = (
                torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
                + graph_data.stems[:, 0])
            if self.version == 'v1' or self.version == 'v4':
                stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
            elif self.version == 'v2' or self.version == 'v3':
                stem_out_cat = torch.cat([out[stem_block_batch_idx],
                                          graph_data.stemtypes,
                                          vec_data[graph_data.stems_batch]], 1)

            stem_preds = self.stem2pred(stem_out_cat)
        else:
            stem_preds = None

        # Fwd per bond
        per_jbond_out = None
        if hasattr(graph_data, "bonds") and do_bonds:
            # Expecting bond in stem format list of [[block, atom]]

            bond_block_batch_idx = (
                torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.bonds_batch]
                + graph_data.bonds[:, 0])
            if self.version == 'v4':
                bondstypes = stememb(graph_data.bondstypes)
                bond_out_cat = torch.cat([out[bond_block_batch_idx], bondstypes], 1)
            else:
                raise NotImplemented
            per_jbond_out = self.bond2pred(bond_out_cat)
            # mean pooling of the 2 jbond atom preds
            if per_jbond_out.size(0) > 0:
                per_jbond_out = per_jbond_out.view(graph_data.bonds.size(0) // 2, 2, -1).mean(dim=1)

        mol_out = gnn.global_mean_pool(out, graph_data.batch)
        mol_preds = self.global2pred(mol_out)

        if self.out_per_mol == 2:
            mol_preds2 = self.global2pred2(mol_out)
            mol_preds = torch.cat([mol_preds, mol_preds2], dim=-1)

        return stem_preds, mol_preds, per_jbond_out

    def forward(self, inputs, rnn_hxs, masks, vec_data=None, do_stems=True, **kwargs):
        graph_data = inputs.mol_graph

        stem_preds, mol_preds, per_jbond_out = self.run_model(
            inputs, vec_data, do_stems, do_bonds=True, **kwargs
        )

        max_branches = self._max_branches
        num_blocks = self._num_blocks
        max_blocks = self._max_blocks

        device = stem_preds.device
        dtype = stem_preds.dtype

        # scatter stems
        stems_batch = graph_data.stems_batch
        stems_cnt = torch.tensor(graph_data.__slices__["stems"][:-1])
        clmn_idxs = torch.arange(len(stems_batch)) - \
                    stems_cnt[stems_batch] + \
                    (torch.arange(graph_data.num_graphs) * max_branches)[stems_batch]
        add_logits = torch.zeros(graph_data.num_graphs, max_branches, num_blocks,
                                 device=device, dtype=dtype)
        add_logits.view(-1, num_blocks)[clmn_idxs.to(device)] = stem_preds

        # scatter bonds
        bonds_batch = graph_data.bonds_batch[::2]
        bonds_cnt = torch.tensor(graph_data.__slices__["bonds"][:-1]) // 2
        clmn_idxs = torch.arange(len(bonds_batch)) - \
                    bonds_cnt[bonds_batch] + \
                    (torch.arange(graph_data.num_graphs) * (max_blocks - 1))[bonds_batch]
        rem_logits = torch.zeros(graph_data.num_graphs, max_blocks - 1, device=device, dtype=dtype)
        rem_logits.view(-1)[clmn_idxs.to(device)] = per_jbond_out.flatten()
        stop_logit = mol_preds[:, 1:2]

        actor_logits = torch.cat([
            stop_logit,
            rem_logits,
            add_logits.view(graph_data.num_graphs, -1)
        ], dim=1)

        value = mol_preds[:, :1]

        return value, actor_logits, rnn_hxs

    def out_to_policy(self, s, stem_o, mol_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o)
            mol_e = torch.exp(mol_o[:, 0])
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o, do_log=True):
        mol_p, stem_p = self.out_to_policy(s, stem_o, mol_o)
        #print(Z.shape, Z.min().item(), Z.mean().item(), Z.max().item())
        if do_log:
            mol_lsm = torch.log(mol_p + 1e-20)
            stem_lsm = torch.log(stem_p + 1e-20)
        else:
            mol_lsm = mol_p
            stem_lsm = stem_p

        #print(mol_lsm.shape, mol_lsm.min().item(), mol_lsm.mean().item(), mol_lsm.max().item())
        #print(stem_lsm.shape, stem_lsm.min().item(), stem_lsm.mean().item(), stem_lsm.max().item(), '--')
        return -self.index_output_by_action(s, stem_lsm, mol_lsm, a)

    def index_output_by_action(self, s, stem_o, mol_o, a):
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        return (
            stem_o[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o
