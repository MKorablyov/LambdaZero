import cv2
import collections
import os.path as osp
from argparse import Namespace
import time
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from typing import List, Any, Tuple
from dataclasses import dataclass
from torch_geometric.data import Batch

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from LambdaZero.utils import get_external_dirs
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.examples.gflow.utils import LogMetrics, TimeStats


datasets_dir, programs_dir, summaries_dir = get_external_dirs()


@dataclass
class TrainBatch:
    parents: Batch  # parents graph batch
    parents_idx: torch.Tensor  # The index of each parent in the batch
    act: torch.Tensor  # action from parent to state
    r: torch.Tensor
    state: Batch  # state graph batch
    done: torch.Tensor  # final state or not (float)
    mols: Tuple  # [Parent mols, state mols

    def __iter__(self):
        return iter([
            self.parents, self.parents_idx, self.act, self.r, self.state, self.done, self.mols
        ])


@dataclass
class Transition:
    pm: Tuple[BlockMoleculeDataExtended]
    pa: Tuple[Tuple[int, int]]
    r: float
    s: BlockMoleculeDataExtended
    d: bool

    def __iter__(self):
        return iter([self.pm, self.pa, self.r, self.s, self.d])


class DataGenerator:
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        self.args = args

        # Get some parameters
        floatX = getattr(args, "floatX", "float64")
        self.floatX = torch.float64 if floatX == 'float64' else torch.float32
        self.min_blocks = getattr(args, 'min_blocks', 2)
        self.max_blocks = getattr(args, 'max_blocks', 7)
        self.reward_exp = getattr(args, 'reward_exp', 1)
        self.reward_norm = getattr(args, 'reward_norm', 1)
        self.random_action_prob = getattr(args, 'random_action_prob', 0)
        self.random_action_temp = getattr(args, 'random_action_temp', 1.)
        self.R_min = getattr(args, 'R_min', 1e-8)
        self.min_info_dict = getattr(args, 'R_min', 1e-8)
        self.r_offset = getattr(args, 'r_offset', 4)
        self.dock_clip_max = getattr(args, 'dock_clip_max', 0)
        self.do_wrong_thing = getattr(args, 'do_wrong_thing', False)
        self.target_norm = getattr(args, "target_norm", [-8.6, 1.10])
        self._log_min_new = getattr(args, "log_min_new", 128)
        self._iter_sample_new_traj = getattr(args, "iter_sample_new_traj", 16)
        self._iter_sample_online = getattr(args, "iter_sample_online", 0)
        self.max_online_mols = getattr(args, "max_online_mols", 1000)
        self.sampled_mols_max = getattr(args, "sampled_mols_max", 5 * 10 ** 4)

        iter_sample_new_online_split = getattr(args, "iter_sample_new_online_split", 1)

        # Check if we want to combine in the training batch samples backward from the online mol
        # buffer
        if self._iter_sample_online != 0:
            assert iter_sample_new_online_split == 1, \
                "Use either iter_sample_online or iter_sample_new_online_split"
        elif iter_sample_new_online_split != 1:
            assert 0 < iter_sample_new_online_split < 1, "not(0 < iter_sample_new_online_split < 1)"
            new_smpl = self._iter_sample_new_traj
            self._iter_sample_new_traj = int(np.round(new_smpl * iter_sample_new_online_split))
            assert self._iter_sample_new_traj > 0, "No new samples"
            self._iter_sample_online = new_smpl - self._iter_sample_new_traj

        self._device = device
        self.sampling_model_prob = sample_prob
        self.sampling_model = model
        self.proxy_reward = proxy

        # ==========================================================================================
        if self.random_action_temp != 1.:
            assert self.random_action_prob == 0

        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(int(time.time()))
        self.test_mols = []
        self.train_mols_map = {}
        self.seen_molecules = set()
        self._last_ep_sampled_mol = 0

        #######
        # This is the "result", here a list of (reward, BlockMolDataExt, info...) tuples
        self.sampled_mols = []
        self.new_sampled_mols = []  # used for logging new
        self.sampled_mols_cnt = 0
        self.online_mols = []
        self.train_transition_buffer = []  # type: List[Transition]
        self.transition_max_priority = 0.000001
        self.train_transitions_priorities = []

        # Some logging
        self._log_metrics_k = ["train_sampled_inv_r"]
        self._metrics = LogMetrics(self._log_metrics_k)

        # ==========================================================================================
        # MDP LOAD
        bpath = osp.join(datasets_dir, getattr(args.mdp_init, "bpath", "fragdb/blocks_PDB_105.json"))
        self.mdp = MolMDPExtended(bpath)  # Used for generating representation
        mdp_init = getattr(args, "mdp_init", {"repr_type": "atom_graph"})
        mdp_init = getattr(mdp_init, "__dict__", mdp_init)
        mdp_init["device"] = device
        self.mdp.post_init(**mdp_init)

        self.mdp.build_translation_table()
        self.ts = [collections.deque(maxlen=100) for _ in range(10)]
        # ==========================================================================================

    @torch.no_grad()
    def get_backward_trajectory(self, mol: BlockMoleculeDataExtended, reward: float):
        # Sample trajectories by walking backwards from the molecules in our dataset

        r = reward
        done = 1
        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action, unless the molecule is already
        # a "terminal" node (if it has no stems, no actions).
        if len(mol.stems):
            samples.append(Transition((mol,), ((-1, 0),), r, mol, done))
            r = done = 0

        while len(mol.blocks):  # and go backwards
            parents, actions = zip(*self.mdp.parents(mol))
            samples.append(Transition(parents, actions, r, mol, done))
            r = done = 0
            backward_parent_idx = self.train_rng.randint(len(parents))
            mol._backward_parent_idx = backward_parent_idx # So we know how to reconstruct traj
            mol = parents[backward_parent_idx]

        return samples[::-1]

    def set_sampling_model(self,
                           model: torch.nn.Module = None, proxy: BaseProxy = None,
                           sample_prob: float = 0.5):
        self.sampling_model = model
        self.sampling_model_prob = sample_prob
        self.proxy_reward = proxy

    def _add_new_trajectory(self, traj):
        """ add only new (forward) sampled trajectory (from model) """

        sample, (r, m, trajectory_stats, inflow, info) = traj
        smpl_data = [r, m, trajectory_stats, inflow, info]
        self.sampled_mols.append(smpl_data)
        self.new_sampled_mols.append(smpl_data)
        self.sampled_mols_cnt += 1
        if len(self.sampled_mols) > self.sampled_mols_max:
            cut_sampled = int(self.sampled_mols_max * 0.2)
            self.sampled_mols = self.sampled_mols[cut_sampled:]

        self._add_mol_to_online(r, r, m)
        self._add_new_transitions(sample)

    def _sample_batch_trajectories(self, num_traj: int):
        """ Sample new trajectories from the model """
        batch = []
        for i in range(num_traj):
            traj = self.get_sample_model()
            self._add_new_trajectory(traj)

            batch.append(traj)
        return batch

    def _add_backward_trajectories(self, num_traj: int):
        """ Sample new trajectories from the model """
        batch = []

        for i in range(num_traj):
            r, score, mol = random.choice(self.online_mols)
            traj = self.get_backward_trajectory(mol, r)
            self._add_new_transitions(traj)
            batch.append(traj)

        return batch

    def run_train_batch(self, trainer, epoch: int, train: bool = True):
        """
            trainer  # type: BasicTrainer

            Run 1 step train loop
            1. Collect new trajectory samples from model: Extend transition replay buffer
            2. Add new trajectories going backward from stored molecules (online_mols) Extend transition replay buffer
            3. Sample from transition replay buffer
            4. Update transition replay buffer
        """
        self._trainer = trainer  # not nice

        logs = {}

        batch_size_in_traj, batch_size = trainer.train_batch_info()

        self._before_sample_train_batch(batch_size_in_traj, batch_size)
        add_new = self.do_add_new_transitions()

        # 1. Sample new set of trajectories from model
        if add_new:
            self._sample_batch_trajectories(self._iter_sample_new_traj)

        # 2. Add new trajectories going backward
        if add_new:
            self._add_backward_trajectories(self._iter_sample_online)

        # 3. Sample training transition batch
        self._before_train_transition_sample()
        samples, sample_idxs = self._sample_train_transitions(batch_size_in_traj, batch_size)

        # Bet batched samples
        batch = self.sample2batch(samples)

        if train:
            train_info, logs = trainer.train_epoch(epoch, batch)

            # 4. Update transition replay buffer (for prioritized sampling)
            self._update_train_batch_priority(sample_idxs, train_info)

        return logs

    def _before_train_transition_sample(self):
        pass

    def do_add_new_transitions(self):
        return True

    def _update_train_batch_priority(self, sample_idxs, train_info):
        pass

    def _before_sample_train_batch(self, batch_size_in_traj: bool, batch_size: int):
        pass

    def _sample_train_transitions(self, batch_size_in_traj: bool, batch_size: int):
        """ Get batch_size of last trajectories or last transitions (batch_size_in_traj True/False)
        """
        buffer = self.train_transition_buffer
        # sample batch of transitions
        # Newest new transitions

        if batch_size_in_traj:
            if batch_size is None or batch_size == 0:
                batch_size = self._iter_sample_new_traj

            # Assume trajectories were introduced in full and in order
            traj_cnt, transition_pos = 0, len(buffer) - 1
            while traj_cnt < batch_size:
                transition_pos -= 1  # Move before done
                while buffer[transition_pos].d != 1:  # while not done
                    transition_pos -= 1
                traj_cnt += 1

            samples = buffer[(transition_pos + 1):]

            return samples, list(range(transition_pos + 1, len(buffer)))
        else:
            sample_idx = list(range(max(len(buffer)-batch_size, 0), len(buffer)))
            return [buffer[ix] for ix in sample_idx], sample_idx

    @torch.no_grad()
    def _sample_model(self, mols: List[BlockMoleculeDataExtended], no_stop=False):
        """ Sample actions for a list of molecules """
        s = self.mdp.mols2batch([self.mdp.mol2repr(x) for x in mols])
        s_o, m_o = self.sampling_model(s)

        if no_stop:
            m_o = m_o * 0 - 1000  # prevent assigning prob to stop when we can't stop

        logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
        if self.random_action_temp != 1:
            logits = logits / self.random_action_temp
        cat = torch.distributions.Categorical(logits=logits)
        action = cat.sample().item()

        return action

    @torch.no_grad()
    def get_sample_model(self, explore=True):
        max_blocks = self.max_blocks
        random_act_p = self.random_action_prob

        m = BlockMoleculeDataExtended()
        samples = []
        trajectory_stats = []

        for t in range(max_blocks):  # TODO: max blocks config
            rand = False
            if explore and random_act_p > 0 and self.train_rng.uniform() < random_act_p:
                num_act = max(1, len(m.stems)) * self.mdp.num_blocks + 1
                action = self.train_rng.randint(int(t < self.min_blocks), num_act)
                rand = True
            else:
                action = self._sample_model([m], no_stop=(t < self.min_blocks))

            if t >= self.min_blocks and action == 0:
                r, info = self._get_reward(m)
                m.action = action
                samples.append(Transition((m,), ((-1, 0),), r, m, True))
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m_old = m

                m.action = action
                m = self.mdp.add_block_to(m, *action)
                m.to_action = action
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r, info = self._get_reward(m)
                    samples.append(Transition(*zip(*self.mdp.parents(m)), r, m, True))
                    break
                else:
                    samples.append(Transition(*zip(*self.mdp.parents(m)), 0, m, False))

        r = float(r)

        return samples, [r, m, trajectory_stats, None, info]

    def _add_mol_to_online(self, r: float, score: float, m: BlockMoleculeDataExtended):
        self.online_mols.append((r, score, m))
        if len(self.online_mols) > self.max_online_mols:
            self.online_mols = sorted(self.online_mols, key=lambda x: x[1])
            self.online_mols = self.online_mols[int(self.max_online_mols * 0.2):]

    def _add_new_transitions(self, samples, priority: float = None):
        # If new priority
        if priority is None:
            priority = self.transition_max_priority
        else:
            self.transition_max_priority = max(priority, self.transition_max_priority)

        # Add training Transition
        self.train_transition_buffer.extend(samples)
        self.train_transitions_priorities.extend([priority] * len(samples))

        # should have here method to cut train_transitions
        max_buffer = self.args.max_transition_buffer_size
        if len(self.train_transition_buffer) > max_buffer:
            self.train_transition_buffer = self.train_transition_buffer[int(max_buffer * 0.2):]
            self.train_transitions_priorities = self.train_transitions_priorities[int(max_buffer * 0.2):]

    @torch.no_grad()
    def _get_reward(self, m: BlockMoleculeDataExtended) -> Tuple[float, dict]:
        rdmol = m.mol
        if rdmol is None:
            return self.R_min, {}
        smi = m.smiles

        # Second Gflow might have different R (different loss based on Gflow1)
        # if smi in self.train_mols_map:
        #     return self.train_mols_map[smi]
        
        res_scores, infos = self.proxy_reward([m])
        info = infos[0]
        r, info = self.train_mols_map[smi] = self.r2r(dockscore=res_scores[0]), info
        return r, info

    def sample2batch(self, samples: List[Transition]) -> TrainBatch:
        """ Input get list of transitions """
        p, a, r, s, d = list(zip(*samples))

        mols = (p, s)
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []),
                               device=self._device).long()
        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch
        flat_list = [item for sublist in p for item in sublist]

        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in flat_list])
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])

        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)

        return TrainBatch(p, p_batch, a, r, s, d, mols)

    def r2r(self, dockscore=None, normscore=None):
        if dockscore is not None:
            normscore = self.r_offset - (np.clip(dockscore, None, self.dock_clip_max) - self.target_norm[0]) / self.target_norm[1]
        normscore = np.clip(normscore, self.R_min, None)  # max(self.R_min, normscore)
        return (normscore/self.reward_norm) ** self.reward_exp

    def inv_r2r(self, r_scores):
        normscore = r_scores ** (1 / self.reward_exp) * self.reward_norm
        dockscore = ((self.r_offset - normscore) * self.target_norm[1]) + self.target_norm[0]
        return dockscore

    def get_stats(self, epoch: int) -> Tuple[dict, List[dict]]:
        """ Return stats about new training molecules & new molecule infos to log (e.g. for topk) """
        # New molecules
        metrics = self._metrics

        new_mol_infos = []
        new_molecules = self.new_sampled_mols
        if len(new_molecules) > 0:
            new_mol_infos = list([x[-1] for x in new_molecules])
            self.new_sampled_mols = []
            # We should filter out here if there are any molecules that we do not want to send to
            # TopK - This will influence logging step of topK

        log_metrics = self._metrics.get_stats(self._log_metrics_k)
        log_metrics["train_num_sampled_mols"] = self.sampled_mols_cnt

        if not epoch % 100:
            last_mols = self.sampled_mols[-1000:]
            last_mol_r = [x[0] for x in last_mols]
            last_mol_s = [x[-1]["smiles"] for x in last_mols]
            other_stats = {
                "train_sampled_mols_R_mean": np.mean(last_mol_r),
                "train_sampled_mols_R_max": np.max(last_mol_r),
                "train_sampled_mols_R_90": np.percentile(last_mol_r, 90),
                "train_sampled_mols_uniq": len(set(last_mol_s)),
            }
            log_metrics.update(other_stats)

        return log_metrics, new_mol_infos

    def stop_samplers_and_join(self):
        print("nothing to stop")
        pass