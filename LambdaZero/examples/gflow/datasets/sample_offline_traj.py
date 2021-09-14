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


class BatchWithOfflineTraj(DataGenerator):
    """ Debug sampler to check influence of offline data """
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        self._batch_offline_num_traj = num_offline = getattr(args, "offline_traj_sample", 0)
        if num_offline > 0:
            assert num_offline < args.iter_sample_new_traj, "Must also sample some fwd trajectories"

            args.iter_sample_new_traj = args.iter_sample_new_traj - num_offline

        # configure max_transition_buffer_size based on new iter_sample_new_traj
        args.max_transition_buffer_size = \
            args.max_transition_buffer_size * args.iter_sample_new_traj + num_offline

        print(f"KEEPING {args.max_transition_buffer_size} trajectories in buffer")

        super(BatchWithOfflineTraj, self).__init__(args, device, model, proxy, sample_prob)

        self.train_transition_buffer_num_traj = 0

    def _sample_train_transitions(self, batch_size_in_traj: bool, batch_size: int):
        """ Get last trajectories or last transitions (batch_size number)"""
        buffer = self.train_transition_buffer
        # sample batch of transitions
        # Newest new transitions

        if batch_size_in_traj:
            if batch_size is None or batch_size == 0:
                batch_size = self._iter_sample_new_traj
            else:
                raise NotImplemented

            # Assume trajectories were introduced in full and in order
            # Add last sampled traj
            traj_cnt, transition_pos = 0, len(buffer) - 1
            while traj_cnt < batch_size:
                transition_pos -= 1  # Move before done
                while buffer[transition_pos].d != 1:  # while not done
                    transition_pos -= 1
                traj_cnt += 1

            samples = buffer[(transition_pos + 1):]
            samples_idx = list(range(transition_pos + 1, len(buffer)))
            new_pos = transition_pos + 1

            # Add trajectories from beginning of buffer
            traj_cnt, transition_pos = 0, -1
            while traj_cnt < self._batch_offline_num_traj and transition_pos < len(buffer):
                transition_pos += 1  # Move before done
                while transition_pos < len(buffer) and buffer[transition_pos].d != 1:  # while not done
                    transition_pos += 1
                traj_cnt += 1

            samples += buffer[:transition_pos+1]
            samples_idx += list(range(0, transition_pos+1))
            return samples, samples_idx
        else:
            raise NotImplemented

    def _add_new_transitions(self, samples, priority: float = None):
        """ Customize to keep max trajectories not max transitions """
        # If new priority
        if priority is None:
            priority = self.transition_max_priority
        else:
            self.transition_max_priority = max(priority, self.transition_max_priority)

        # Add training Transition
        self.train_transition_buffer.extend(samples)
        self.train_transitions_priorities.extend([priority] * len(samples))

        # should have here method to cut train_transitions
        if self.train_transition_buffer_num_traj >= self.args.max_transition_buffer_size:
            transition_pos = 0
            while self.train_transition_buffer[transition_pos].d != 1:  # while not done
                transition_pos += 1

            transition_pos += 1

            self.train_transition_buffer = self.train_transition_buffer[transition_pos:]
            self.train_transitions_priorities = self.train_transitions_priorities[transition_pos:]
        else:
            self.train_transition_buffer_num_traj += 1

