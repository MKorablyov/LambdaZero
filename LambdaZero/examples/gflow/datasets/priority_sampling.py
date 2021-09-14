import cv2
from argparse import Namespace
import numpy as np
import torch
import random

from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class PrioritySamplingData(DataGenerator):
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        super(PrioritySamplingData, self).__init__(args, device, model, proxy, sample_prob)
        self.priority_key = getattr(args, "priority_key", "losses")
        self.priority_get_max = getattr(args, "priority_get_max", False)
        self.true_priority = getattr(args, "priority_true", True)
        self.priority_add_new_freq = getattr(args, "priority_add_new_freq", 4)
        self.add_new_req = 0

    def _add_new_transitions(self, samples, priority: float = None):
        """ Should keep highest priority transitions """

        if priority is None:
            priority = self.transition_max_priority
            if self.true_priority:
                batch = self.sample2batch(samples)
                priority = self._trainer.compute_loss(batch).data.cpu().numpy().tolist()
                self.transition_max_priority = max(max(priority), self.transition_max_priority)
        else:
            self.transition_max_priority = max(priority, self.transition_max_priority)

        if not hasattr(priority, "__len__"):
            priority = [priority] * len(samples)

        # Add training Transition
        self.train_transition_buffer.extend(samples)
        self.train_transitions_priorities.extend(priority)

        # should have here method to cut train_transitions
        max_buffer = self.args.max_transition_buffer_size
        if len(self.train_transition_buffer) > max_buffer:
            srt_idxs = np.argsort(self.train_transitions_priorities)[int(max_buffer * 0.2):]
            self.train_transition_buffer = [self.train_transition_buffer[ix] for ix in srt_idxs]
            self.train_transitions_priorities = [self.train_transitions_priorities[ix] for ix in srt_idxs]
            self.transition_max_priority = self.train_transitions_priorities[-1]

    def _sample_train_transitions(self, batch_size_in_traj: bool, batch_size: int):
        """
            Sampling should not be such a big problem for small buffer.
            We are expecting to train the Gflow not so off-policy
        """
        assert batch_size > 0, "NO NO"

        # sample batch of transitions
        buffer = self.train_transition_buffer
        priority = self.train_transitions_priorities
        max_p = self.transition_max_priority

        sample_idx = []
        if self.priority_get_max:
            idx = len(priority) - 1
            while priority[idx] >= max_p and len(sample_idx) < batch_size and idx >= 0:
                sample_idx.append(idx)
                idx -= 1

            if len(sample_idx) < batch_size and idx > 0:
                remaining_priority = priority[:idx]
                r_idx = random.choices(range(len(remaining_priority)),
                                       weights=remaining_priority,
                                       k=batch_size-len(sample_idx))
                sample_idx += r_idx
        else:
            # sample batch of transitions
            sample_idx = random.choices(range(len(buffer)), weights=priority, k=batch_size)

        return [buffer[ix] for ix in sample_idx], sample_idx

    def _update_train_batch_priority(self, sample_idxs, train_info):
        new_priorities = train_info[self.priority_key].data.cpu().numpy()

        for idx, priority in zip(sample_idxs, new_priorities):
            self.train_transitions_priorities[idx] = priority

        self.transition_max_priority = max(self.train_transitions_priorities)

    def do_add_new_transitions(self):
        # Add when we dont want
        add_new = False
        if self.add_new_req % self.priority_add_new_freq == 0:
            add_new = True

        self.add_new_req += 1
        return add_new
