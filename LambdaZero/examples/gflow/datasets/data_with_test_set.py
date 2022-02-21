import argparse
import os

import cv2
from argparse import Namespace
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
import random
from argparse import Namespace
from typing import List, Any

from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator, TrainBatch
from LambdaZero.examples.gflow.datasets.sample_1transition import parents_to_del_stem
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class DataWithTestSet(DataGenerator):
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        super(DataWithTestSet, self).__init__(args, device, model, proxy, sample_prob)

        self._sample_traj_bwd_trans = getattr(self.args, "sample_traj_bwd_trans", False)
        self._tr_balance = getattr(self.args, "tr_balance", False)

        self._buffer_traj_dump = 0
        self._buffer_traj_dump_freq = 100
        self._out_dir = args.out_dir

        self._hold_out_test_set = args.hold_out_test_set
        self._eval_epoch_freq = args.eval_epoch_freq

        # For eval
        self._test_logs = Namespace()
        self._test_logs.skipped_traj = defaultdict(int)
        self._test_logs.skipped_traj_count = 0
        self._buffer_traj = list()
        self._eval_dump_dir = f"{self._out_dir}/eval"

        test_set = np.load(args.test_set, allow_pickle=True).item()
        test_df = test_set["all_smiles"]
        self._test_set_smiles = set(test_df)
        del test_df

        if not os.path.isdir(self._eval_dump_dir):
            try:
                os.mkdir(self._eval_dump_dir)
            except:
                print(f'Could not create folder {self._eval_dump_dir}')

    def _sample_batch_trajectories(self, num_traj: int):
        hold_out_test_set = self._hold_out_test_set
        test_set_smiles = self._test_set_smiles

        batch = []
        store_batch = []
        while len(batch) < num_traj:
            _batch = super()._sample_batch_trajectories(num_traj - len(batch))

            if self._hold_out_test_set:
                for samples, [r, m, trajectory_stats, inflow, info] in _batch:
                    smi = m.smiles
                    if smi not in test_set_smiles:
                        batch.append([samples, [r, m, trajectory_stats, inflow, info]])
                        store_batch.append([
                            [x.s.dump() for x in samples], r, m.dump(), trajectory_stats, info
                        ])
                    else:
                        self._test_logs.skipped_traj[smi] += 1
                        self._test_logs.skipped_traj_count += 1
            else:
                batch += _batch

        self._buffer_traj.append(store_batch)

        # if len(self._buffer_traj) > self._buffer_traj_dump_freq:
        #     np.save(f"{self._traj_dump_dir}/train_traj_dump_{self._buffer_traj_dump}", self._buffer_traj)
        #     self._buffer_traj.clear()
        #     self._buffer_traj_dump += 1
        #
        return batch

    def _add_new_trajectory(self, traj):
        # Filter adding trajectories from test set
        sample, (r, m, trajectory_stats, inflow, info) = traj

        if not self._hold_out_test_set or m.smiles not in self._test_set_smiles:
            super()._add_new_trajectory(traj)

    def _add_mol_to_online(self, r, score, m):
        # Filter adding molecules from test set
        if not self._hold_out_test_set or m.smiles not in self._test_set_smiles:
            super()._add_mol_to_online(r, score, m)

    def run_train_batch(self, trainer, epoch: int, train: bool = True):
        logs = super().run_train_batch(trainer, epoch, train)

        if epoch > 0 and (epoch % self._eval_epoch_freq == 0 or epoch == 100):
            # Save stuff
            torch.save({
                "epoch": epoch,
                "logs": logs,
                "model": trainer.model.state_dict(),
            }, f"{self._eval_dump_dir}/ep{epoch}_model.pk")

            torch.save({
                "epoch": epoch,
                "logs": logs,
                "train_traj": self._buffer_traj,
                "test_logs": self._test_logs,
            }, f"{self._eval_dump_dir}/ep{epoch}_eval_logs.pk")

            self._buffer_traj.clear()

        logs["skipped_traj_count"] = self._test_logs.skipped_traj_count
        return logs

    def sample2batch(self, samples: List[Any]):
        if not self._tr_balance:
            return super().sample2batch(samples)

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

