from argparse import Namespace
from typing import List, Any
import numpy as np
import torch
from collections import Counter

from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator, TrainBatch
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.utils import get_external_dirs
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


def parents_to_del_stem(parents: List[BlockMoleculeDataExtended], mol: BlockMoleculeDataExtended):
    # Empty block parent of action stop
    if len(mol.jbonds) <= 0 or parents[0][1][0] == -1:
        return np.array([[]]).transpose().reshape(-1, 2)

    del_stems = []
    for ixp, (parent, (block, parent_stem)) in enumerate(parents):
        connect_block, connect_block_atm = parent.stems[parent_stem]

        # Add atm idx to which this new block is connected (in mol representation)
        connect_block_idx = mol._parent_connected_block[ixp]
        del_stems.append([connect_block_idx, connect_block_atm])

    return np.array(del_stems).reshape(-1, 2)


class OnlineDataFeedTransition(DataGenerator):
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        super(OnlineDataFeedTransition, self).__init__(args, device, model, proxy, sample_prob)
        self._sample_traj_bwd_trans = getattr(args, "sample_traj_bwd_trans", False)

    # def _sample_train_transitions(self, batch_size: int):
    #     """ Train only on last transition """
    #     # sample batch of transitions
    #     samples = []
    #     traj_cnt = 0
    #     tr = self.train_transition_buffer
    #     transition_pos = len(tr) - 1
    #
    #     while traj_cnt < self.args.iter_sample_new_traj:
    #         transition_pos -= 1  # Move before done
    #         while tr[transition_pos].d != 1:  # while not done
    #             transition_pos -= 1
    #         traj_cnt += 1
    #
    #     samples = tr[(transition_pos+1):]
    #
    #     return samples, list(range(transition_pos+1, len(tr)))

    def sample2batch(self, samples: List[Any]):
        atom_graph = self.mdp.repr_type == "atom_graph"

        n_samples = []
        bonds_atmidxs = []
        bonds_slices = [0]
        bonds_batch = []
        bonds_blocks = []
        bonds_count = 0
        for ix, (p, a, r, s, d) in enumerate(samples):
            if self._sample_traj_bwd_trans:
                # Choose one parent at random
                idx = self.train_rng.randint(len(p))
            else:
                # Choose the true parent that produced the trajectory
                if len(p) == 1:
                    idx = 0
                else:
                    if hasattr(s, "_backward_parent_idx"):
                        idx = s._backward_parent_idx
                    else:
                        # Should be the one that was last (traj constructed by adding new block at end)
                        idx = s._parent_deleted_block.index(len(s.blockidxs) - 1)

            parent_mol = p[idx]
            fwd_act = a[idx]

            # reorder
            parents, acts = list(p), list(a)
            sorted_p = [parents.pop(idx)] + parents
            sorted_a = [acts.pop(idx)] + acts
            del_stems = parents_to_del_stem(list(zip(sorted_p, sorted_a)), s)

            if atom_graph:
                jbond_atmidx = np.array(s.slices)[del_stems[:, 0]] + del_stems[:, 1]
                jbond_atmidx = jbond_atmidx[:, None]
            else:
                blockidxs = s.blockidxs
                bonds_blocks += [blockidxs[ix] for ix in del_stems[:, 0]]
                jbond_atmidx = del_stems

            n_samples.append(((parent_mol,), (fwd_act,), r, s, d))

            bonds_atmidxs.append(jbond_atmidx)
            bonds_count += len(jbond_atmidx)
            bonds_batch += [ix] * len(jbond_atmidx)
            bonds_slices.append(bonds_count)

        (p, p_batch, a, r, state, d, mols) = super().sample2batch(n_samples)

        bonds_atmidxs = np.concatenate(bonds_atmidxs)
        state.bonds = torch.tensor(bonds_atmidxs, device=self._device).long()
        state.bonds_batch = torch.tensor(bonds_batch, device=self._device).long()
        state.__slices__['bonds'] = bonds_slices
        if not atom_graph:
            # bond_graph = so we should construct bond_stem_types
            true_blockidx = self.mdp.true_blockidx
            stem_type_offset = self.mdp.stem_type_offset
            stemtypes = [
                stem_type_offset[true_blockidx[bonds_blocks[i]]] + bonds_atmidxs[i, 1]
                for i in range(len(bonds_atmidxs))
            ]
            state.bondstypes = torch.tensor(stemtypes, device=self._device).long()

        return TrainBatch(p, p_batch, a, r, state, d, mols)
