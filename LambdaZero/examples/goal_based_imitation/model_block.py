
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

class GraphAgent(nn.Module):

    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v1'):
        super().__init__()
        self.version = version
        self.embeddings = nn.ModuleList([
            nn.Embedding(mdp_cfg.num_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)])
        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean')
        nvec_1 = nvec * (version == 'v1')
        nvec_2 = nvec * (version == 'v2')
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb))
        self.gru = nn.GRU(nemb, nemb)
        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))
        self.global2pred = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                         nn.Linear(nemb, out_per_mol))
        #self.set2set = Set2Set(nemb, processing_steps=3)
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'escort'
        self.escort_p = 4


    def forward(self, graph_data, vec_data):
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x)
        graph_data.stemtypes = stememb(graph_data.stemtypes)
        graph_data.edge_attr = bondemb(graph_data.edge_attr)
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2))
        out = graph_data.x
        if self.version == 'v1':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        elif self.version == 'v2':
            out = self.block2emb(out)

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        stem_block_batch_idx = (
            torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
            + graph_data.stems[:, 0])
        if self.version == 'v1':
            stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
        elif self.version == 'v2':
            stem_out_cat = torch.cat([out[stem_block_batch_idx],
                                      graph_data.stemtypes,
                                      vec_data[graph_data.stems_batch]], 1)

        stem_preds = self.stem2pred(stem_out_cat)
        mol_preds = self.global2pred(gnn.global_mean_pool(out, graph_data.batch))
        return stem_preds, mol_preds

    def out_to_policy(self, s, stem_o, mol_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o - 2)
            mol_e = torch.exp(mol_o[:, 0] - 2)
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-3
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        mol_p, stem_p = self.out_to_policy(s, stem_o, mol_o)
        #print(Z.shape, Z.min().item(), Z.mean().item(), Z.max().item())
        mol_lsm = torch.log(mol_p + 1e-5)
        stem_lsm = torch.log(stem_p + 1e-5)
        #print(mol_lsm.shape, mol_lsm.min().item(), mol_lsm.mean().item(), mol_lsm.max().item())
        #print(stem_lsm.shape, stem_lsm.min().item(), stem_lsm.mean().item(), stem_lsm.max().item(), '--')
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_lsm.device)
        return -(
            stem_lsm[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_lsm * (a[:, 0] == -1))


def mol2graph(mol, mdp):
    f = lambda x: torch.tensor(x, dtype=torch.long, device=mdp.device)
    if len(mol.blockidxs) == 0:
        data = Data(# There's an extra block embedding for the empty molecule
            x=f([mdp.num_blocks]),
            edge_index=f([[],[]]),
            edge_attr=f([]).reshape((0,2)),
            stems=f([(0,0)]),
            stemtypes=f([mdp.num_stem_types])) # also extra stem type embedding
        return data
    edges = [(i[0], i[1]) for i in mol.jbonds]
    #edge_attrs = [mdp.bond_type_offset[i[2]] +  i[3] for i in mol.jbonds]
    if 0:
        edge_attrs = [((mdp.stem_type_offset[mol.blockidxs[i[0]]] + i[2]) * mdp.num_stem_types +
                       (mdp.stem_type_offset[mol.blockidxs[i[1]]] + i[3]))
                      for i in mol.jbonds]
    else:
        edge_attrs = [(mdp.stem_type_offset[mol.blockidxs[i[0]]] + i[2],
                       mdp.stem_type_offset[mol.blockidxs[i[1]]] + i[3])
                      for i in mol.jbonds]
    stemtypes = [mdp.stem_type_offset[mol.blockidxs[i[0]]] + i[1] for i in mol.stems]

    data = Data(x=f(mol.blockidxs),
                edge_index=f(edges).T if len(edges) else f([[],[]]),
                edge_attr=f(edge_attrs) if len(edges) else f([]).reshape((0,2)),
                stems=f(mol.stems) if len(mol.stems) else f([(0,0)]),
                stemtypes=f(stemtypes) if len(mol.stems) else f([mdp.num_stem_types]))
    data.to(mdp.device)
    #print(data)
    return data


def mols2batch(mols, mdp):
    batch = Batch.from_data_list(
        mols, follow_batch=['stems'])
    batch.to(mdp.device)
    return batch
