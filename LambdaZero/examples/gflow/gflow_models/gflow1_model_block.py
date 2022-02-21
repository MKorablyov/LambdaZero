import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from argparse import Namespace
from LambdaZero.examples.gflow.gflow_models.gflow_model_base import GFlowModelBase


class GraphAgent(GFlowModelBase):
    def __init__(self, cfg: Namespace, **kwargs):
        super(GraphAgent, self).__init__(cfg, **kwargs)

        nemb = getattr(cfg, "nemb", 256)
        nvec = getattr(cfg, "nvec", 0)
        out_per_stem = getattr(cfg, "out_per_stem", 105)
        self.out_per_mol = out_per_mol = getattr(cfg, "out_per_mol", 1)
        num_conv_steps = getattr(cfg, "num_conv_steps", 10)
        version = getattr(cfg, "version", "v4")
        repr_type = getattr(cfg, "repr_type", "block_graph")
        mdp_cfg = kwargs["mdp"]
        assert repr_type == "block_graph", "GraphAgent works on block_graph input"

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
        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))
        self.bond2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))

        self.global2pred = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                         nn.Linear(nemb, 1))
        if out_per_mol == 2:
            self.global2pred2 = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                              nn.Linear(nemb, 1))
        elif out_per_mol != 1:
            raise NotImplemented

        #self.set2set = Set2Set(nemb, processing_steps=3)
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'softmax'
        self.escort_p = 6

    def run_model(self, graph_data, vec_data=None, do_stems=True, do_bonds=True, **kwargs):
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

        mol_out = gnn.global_mean_pool(out, graph_data.batch)
        mol_preds = self.global2pred(mol_out)

        if self.out_per_mol == 2:
            mol_preds2 = self.global2pred2(mol_out)
            mol_preds = torch.cat([mol_preds, mol_preds2], dim=-1)

        return stem_preds, mol_preds, per_jbond_out

    def forward(self, graph_data, vec_data=None, do_stems=True, ret_values=False, **kwargs):
        stem_preds, mol_preds, _ = self.run_model(
            graph_data, vec_data, do_stems, do_bonds=False, **kwargs
        )
        if ret_values:
            return stem_preds, mol_preds

        return stem_preds, mol_preds[:, :1]

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
