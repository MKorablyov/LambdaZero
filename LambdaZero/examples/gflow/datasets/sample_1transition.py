import pdb
import time

from collections import deque

from argparse import Namespace
from typing import List, Any
import numpy as np
import torch
from torch.multiprocessing import Process, Queue
import os.path as osp
from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended
import random
from LambdaZero.utils import get_external_dirs
from collections import Counter
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended


datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def get_rev_actions(parent: BlockMoleculeDataExtended, act, mol: BlockMoleculeDataExtended):
    # TODO Action stop chosen. Not sure if this is the correct way to handle action stop step
    if len(mol.jbonds) <= 0 or act[0] == -1:
        return np.array([[]]).transpose()

    block_counts = Counter(sum([x[:2] for x in mol.jbonds], []))

    single_block_delete = []
    for ibond, bond in enumerate(mol.jbonds):
        if block_counts[bond[0]] == 1:
            single_block_delete.append((ibond, 0))

        if block_counts[bond[1]] == 1:
            single_block_delete.append((ibond, 1))

    # Calculate inv action
    rev_action = None
    (nblck, nstem) = act
    c_stem = parent.stems[nstem]
    c_block, c_bond = parent.blockidxs[c_stem[0]], c_stem[1]
    for idx, (ibond, _) in enumerate(single_block_delete):
        bond = mol.jbonds[ibond]
        if c_bond == bond[2] and c_block == mol.blockidxs[bond[0]]:
            rev_action = (ibond, 1)
            break
        elif c_bond == bond[3] and c_block == mol.blockidxs[bond[1]]:
            rev_action = (ibond, 0)
            break

    # if rev_action is None:
    #     print("SRC")
    #     import pdb; pdb.set_trace()
    single_block_delete.pop(idx)
    single_block_delete = [rev_action] + single_block_delete

    # Transform reverse actions to jbond_atmidxs
    slices = np.asarray(mol.slices)
    jbonds = np.asarray(mol.jbonds)

    racts = np.array(single_block_delete)
    jidx, joff = racts[:, 0], racts[:, 1]
    sel_jbonds = jbonds[jidx]
    jbond_atmidx = slices[np.take_along_axis(sel_jbonds, joff[:, None], 1)]
    jbond_atmidx += np.take_along_axis(sel_jbonds, (joff[:, None] + 2), 1)

    # First jbond_idx = is the rev action from mol -> parent
    single_block_delete_atmidx = jbond_atmidx
    return single_block_delete_atmidx


class OnlineDataFeedTransition(DataGenerator):
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        super(OnlineDataFeedTransition, self).__init__(args, device, model, proxy, sample_prob)

    def _sample_train_transitions(self, batch_size: int):
        samples = super()._sample_train_transitions(batch_size)

        # sample batch of transitions
        single_tr_samples = []

        for p, a, r, s, d in samples:
            idx = self.train_rng.randint(len(p))

            parent_mol = p[idx]
            fwd_act = a[idx]
            rev_actions = get_rev_actions(parent_mol, fwd_act, s)

            single_tr_samples.append(
                ((parent_mol,), (fwd_act,), r, s, d, rev_actions)
            )

        return single_tr_samples

    def sample2batch(self, samples: List[Any]):
        n_samples = []
        bonds_atmidx = []
        bonds_slices = [0]
        bonds_batch = []
        bonds_count = 0
        for ix, smpl in enumerate(samples):
            n_samples.append(smpl[:-1])
            bonds_atmidx.append(smpl[-1])
            bonds_count += len(smpl[-1])
            bonds_batch += [ix] * len(smpl[-1])
            bonds_slices.append(bonds_count)

        (p, p_batch, a, r, state, d, mols) = super().sample2batch(n_samples)

        bonds_atmidx = np.concatenate(bonds_atmidx)
        state.bonds = torch.tensor(bonds_atmidx, device=self._device).long()
        state.bonds_batch = torch.tensor(bonds_batch, device=self._device).long()
        state.__slices__['bonds'] = bonds_slices

        return (p, p_batch, a, r, state, d, mols)
