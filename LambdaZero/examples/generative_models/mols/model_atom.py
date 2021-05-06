"""
Code for an atom-based graph representation and architecture

"""



import warnings
warnings.filterwarnings('ignore')
import sys
import time
import os
import os.path as osp
import pickle
import gzip
import psutil
import subprocess


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

from LambdaZero import chem
from LambdaZero.chem import atomic_numbers
from LambdaZero.environments.persistent_search import PersistentSearchTree, PredDockRewardActor, SimDockRewardActor, RLActor, MBPrep, RandomRLActor
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

warnings.filterwarnings('ignore')


class MPNNet_v2(nn.Module):
    def __init__(self, num_feat=14, num_vec=3, dim=64,
                 num_out_per_mol=1, num_out_per_stem=105,
                 num_conv_steps=12, version='v1'):
        super().__init__()
        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        assert version in ['v1', 'v2']
        self.version = int(version[1:])

        net = nn.Sequential(nn.Linear(4, 128), nn.LeakyReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.lin1 = nn.Linear(dim, dim * 8)
        self.lin2 = nn.Linear(dim * 8, num_out_per_stem)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, num_out_per_mol)

    def forward(self, data, vec_data=None, do_stems=True):
        if self.version == 1:
            batch_vec = vec_data[data.batch]
            out = F.leaky_relu(self.lin0(torch.cat([data.x, batch_vec], 1)))
        elif self.version == 2:
            out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        if do_stems:
            # Index of the origin atom of each stem in the batch, we
            # need to adjust for the batch packing)
            stem_batch_idx = (
                torch.tensor(data.__slices__['x'], device=out.device)[data.stems_batch]
                + data.stems)
            stem_atom_out = out[stem_batch_idx]
            per_stem_out = self.lin2(F.leaky_relu(self.lin1(stem_atom_out)))
        else:
            per_stem_out = None

        out = self.set2set(out, data.batch)
        sout = self.lin3(out) # per mol scalar outputs
        return sout, per_stem_out


class MolAC_GCN(nn.Module):
    def __init__(self, nhid, nvec, num_out_per_stem, num_out_per_mol, num_conv_steps, version):
        nn.Module.__init__(self)
        act = nn.LeakyReLU()
        self.training_steps = 0

        self.mpnn = MPNNet_v2(
            num_feat=14 + 1 + len(atomic_numbers),
            num_vec=nvec,
            dim=nhid,
            num_out_per_mol=num_out_per_mol,
            num_out_per_stem=num_out_per_stem,
            num_conv_steps=num_conv_steps,
            version=version)

    def out_to_policy(self, s, stem_o, mol_o):
        stem_e = torch.exp(stem_o)
        mol_e = torch.exp(mol_o[:, 0])
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        stem_e = torch.exp(stem_o - 2)
        mol_e = torch.exp(mol_o[:, 0] - 2)
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e
        mol_lsm = torch.log(mol_e / Z)
        stem_lsm = torch.log(stem_e / Z[s.stems_batch, None])
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_lsm.device)
        return -(
            stem_lsm[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_lsm * (a[:, 0] == -1))

    def index_output_by_action(self, s, stem_o, mol_o, a):
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        return -(
            stem_o[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

    def forward(self, graph, vec=None, do_stems=True):
        sout, logits = self.mpnn(graph, vec, do_stems=do_stems)
        return logits, sout

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))

def mol2graph(mol, mdp):
    rdmol = mol.mol
    if rdmol is None:
        g = Data(x=torch.zeros((1, 14 + len(atomic_numbers))),
                 edge_attr=torch.zeros((0, 4)),
                 edge_index=torch.zeros((0, 2)).long())
    else:
        atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False,
                                                    one_hot_atom=True, donor_features=False)
        g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
    stems = mol.stem_atmidxs
    if not len(stems):
        stems = [0]
    stem_mask = torch.zeros((g.x.shape[0], 1))
    stem_mask[torch.tensor(stems).long()] = 1
    g.stems = torch.tensor(stems).long()
    g.x = torch.cat([g.x, stem_mask], 1)
    if g.edge_index.shape[0] == 0:
        g.edge_index = torch.zeros((2, 1)).long()
        g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).float()
        g.stems = torch.zeros((1,)).long()
    return g


def mols2batch(mols, mdp):
    batch = Batch.from_data_list(
        mols, follow_batch=['stems'])
    batch.to(mdp.device)
    return batch
