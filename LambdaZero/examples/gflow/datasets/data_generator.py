import cv2
from copy import deepcopy
import gzip
import os
import os.path as osp
from argparse import Namespace
import time
import random
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
from typing import NamedTuple, Tuple
from typing import List, Any, Tuple

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from LambdaZero.utils import get_external_dirs
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.examples.gflow.utils import LogMetrics, TimeStats

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class Transition(NamedTuple):
    parent_mol: Tuple[BlockMoleculeDataExtended]  # first item should be the true trajectory parent
    parent_act: Tuple[Tuple[int, int]]  # first item should be the true trajectory parent
    r: float
    state: BlockMoleculeDataExtended
    done: bool
    info: dict = dict()


class DataGenerator:
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        self.args = args

        floatX = getattr(args, "floatX", "float64")
        self.floatX = torch.float64 if floatX == 'float64' else torch.float32
        self.min_blocks = getattr(args, 'min_blocks', 2)
        self.max_blocks = getattr(args, 'max_blocks', 7)
        self.reward_exp = getattr(args, 'reward_exp', 1)
        self.reward_norm = getattr(args, 'reward_norm', 1)
        self.random_action_prob = getattr(args, 'random_action_prob', 0)
        self.R_min = getattr(args, 'R_min', 1e-8)
        self.min_info_dict = getattr(args, 'R_min', 1e-8)
        self.r_offset = getattr(args, 'r_offset', 4)
        self.do_wrong_thing = getattr(args, 'do_wrong_thing', False)
        self.target_norm = getattr(args, "target_norm", [-8.6, 1.10])
        self._log_min_new = getattr(args, "log_min_new", 128)
        self._iter_sample_new_traj = getattr(args, "iter_sample_new_traj", 16)
        self.max_online_mols = getattr(args, "max_online_mols", 1000)
        self._device = device
        self.sampling_model_prob = sample_prob
        self.sampling_model = model
        self.proxy_reward = proxy

        self.test_split_rng = np.random.RandomState(142857)
        self.train_rng = np.random.RandomState(int(time.time()))
        self.test_mols = []
        self.train_mols_map = {}
        self.seen_molecules = set()
        self._last_ep_sampled_mol = 0
        #######
        # This is the "result", here a list of (reward, BlockMolDataExt, info...) tuples
        self.sampled_mols = []
        self.online_mols = []
        self.train_transition_buffer = []
        self.train_transition_buffer_scores = []

        self._log_metrics_k = ["train_sampled_inv_r"]
        self._metrics = LogMetrics(self._log_metrics_k)

        # self.mdp = MolMDPExtended(bpath)
        # self.mdp.post_init(device, args.repr_type, include_nblocks=args.include_nblocks)
        # self.mdp.build_translation_table()
        # self.mdp.floatX = self.floatX
        # self.mdp._cue_max_blocks = self.max_blocks  # TODO not sure what for
        # ==========================================================================================
        # MDP LOAD

        bpath = osp.join(datasets_dir, getattr(args.mdp_init, "bpath", "fragdb/blocks_PDB_105.json"))
        self.mdp = MolMDPExtended(bpath)  # Used for generating representation
        mdp_init = getattr(args, "mdp_init", {"repr_type": "atom_graph"})
        mdp_init = getattr(mdp_init, "__dict__", mdp_init)
        mdp_init["device"] = device
        self.mdp.post_init(**mdp_init)

        self.mdp.build_translation_table()

        # ==========================================================================================

    @torch.no_grad()
    def _get(self, i, dset):
        if ((self.sampling_model_prob > 0 and # don't sample if we don't have to
             self.train_rng.uniform() < self.sampling_model_prob)
            or len(dset) < 32):
            return self.get_sample_model()

        # Sample trajectories by walking backwards from the molecules in our dataset
        # print("NEW GET", i, "sample_online")
        # Handle possible multithreading issues when independent threads
        # add/substract from dset:
        while True:
            try:
                m = dset[i]
            except IndexError:
                i = self.train_rng.randint(0, len(dset))
                continue
            break
        if not isinstance(m, BlockMoleculeDataExtended):
            m = m[-1]
        r = m.reward
        done = 1
        samples = []
        # a sample is a tuple (parents(s), parent actions, reward(s), s, done)
        # an action is (blockidx, stemidx) or (-1, x) for 'stop'
        # so we start with the stop action, unless the molecule is already
        # a "terminal" node (if it has no stems, no actions).
        if len(m.stems):
            samples.append(((m,), ((-1, 0),), r, m, done))
            r = done = 0
        while len(m.blocks): # and go backwards
            parents, actions = zip(*self.mdp.parents(m))
            samples.append((parents, actions, r, m, done))
            r = done = 0
            m = parents[self.train_rng.randint(len(parents))]
        return samples

    def set_sampling_model(self,
                           model: torch.nn.Module = None, proxy: BaseProxy = None,
                           sample_prob: float = 0.5):
        self.sampling_model = model
        self.sampling_model_prob = sample_prob
        self.proxy_reward = proxy

    def _sample_new_trajectory(self):
        sample, (r, m, trajectory_stats, inflow, info) = self.get_sample_model()
        self.sampled_mols.append((r, m, trajectory_stats, inflow, info))

        self._add_mol_to_online(r, r, m)
        self._add_new_transitions(sample, r, m)

        return sample, (r, m, trajectory_stats, inflow, info)

    def sample_train_batch(self, batch_size: int):
        """ Sample batch_size number of transitions (get batch from sampled transitions only) """

        # Sample new trajectories from the model
        for i in range(self._iter_sample_new_traj):
            self._sample_new_trajectory()

        # Run before sampling transitions (for e.g. we could sample here new backward trajectories)
        self._before_train_transition_sample()

        # Sample training transition batch
        samples = self._sample_train_transitions(batch_size)

        # return samples
        batch = self.sample2batch(samples)
        return batch

    def _before_train_transition_sample(self):
        pass

    def _sample_train_transitions(self, batch_size: int):
        # sample batch of transitions
        samples = random.choices(self.train_transition_buffer, k=batch_size)
        return samples

    @torch.no_grad()
    def get_sample_model(self):
        m = BlockMoleculeDataExtended()
        samples = []
        actions = []
        max_blocks = self.max_blocks
        trajectory_stats = []

        for t in range(max_blocks):  # TODO: max blocks config
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            s_o, m_o = self.sampling_model(s)

            if t < self.min_blocks:
                m_o = m_o * 0 - 1000  # prevent assigning prob to stop when we can't stop

            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(logits=logits)
            action = cat.sample().item()
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < self.min_blocks), logits.shape[0])

            q = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            trajectory_stats.append((q[action].item(), action, torch.logsumexp(q, 0).item()))
            if t >= self.min_blocks and action == 0:
                r, info = self._get_reward(m)
                actions.append((-1, 0))
                samples.append(((m,), ((-1, 0),), r, m, 1))
                break
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                actions.append(action)
                m_old = m
                m = self.mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r, info = self._get_reward(m)
                    if self.do_wrong_thing:
                        samples.append(((m_old,), (action,), r, m, 1))
                    else:
                        samples.append((*zip(*self.mdp.parents(m)), r, m, 1))
                    break
                else:
                    if self.do_wrong_thing:
                        samples.append(((m_old,), (action,), 0, m, 0))
                    else:
                        samples.append((*zip(*self.mdp.parents(m)), 0, m, 0))

        p = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in samples[-1][0]])
        qp = self.sampling_model(p)
        qsa_p = self.sampling_model.index_output_by_action(
            p, qp[0], qp[1][:, 0],
            torch.tensor(samples[-1][1], device=self._device).long()
        )
        inflow = torch.logsumexp(qsa_p.flatten(), 0).item()
        r = float(r)

        # samples = [Transition(*x, info={"end_score": r}) for x in samples]
        return samples, (r, m, trajectory_stats, inflow, info)

    def _add_mol_to_online(self, r, score, m):
        self.online_mols.append((r, score, m))
        if len(self.online_mols) > self.max_online_mols:
            self.online_mols = self.online_mols[-self.max_online_mols:]

    def _add_new_transitions(self, samples, r, m):
        # Add training Transition
        self.train_transition_buffer.extend(samples)
        self.train_transition_buffer_scores.extend([r] * len(samples))

        # should have here method to cut train_transitions
        max_buffer = self.args.max_transition_buffer_size
        if len(self.train_transition_buffer) > max_buffer:
            self.train_transition_buffer = self.train_transition_buffer[-max_buffer:]
            self.train_transition_buffer_scores = self.train_transition_buffer_scores[-max_buffer:]

    def _get_reward(self, m: BlockMoleculeDataExtended) -> Tuple[float, dict]:
        rdmol = m.mol
        if rdmol is None:
            return self.R_min, {}
        smi = m.smiles
        if smi in self.train_mols_map:
            return self.train_mols_map[smi]
        res_scores, infos = self.proxy_reward([m])
        info = infos[0]
        r, info = self.train_mols_map[smi] = self.r2r(dockscore=res_scores[0]), info
        return r, info

    def sample2batch(self, samples: List[Any]):
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
        return (p, p_batch, a, r, s, d, mols)

    def r2r(self, dockscore=None, normscore=None):
        if dockscore is not None:
            normscore = self.r_offset - (min(0, dockscore)-self.target_norm[0])/self.target_norm[1]
        normscore = max(self.R_min, normscore)
        return (normscore/self.reward_norm) ** self.reward_exp

    def inv_r2r(self, r_scores):
        normscore = r_scores ** (1 / self.reward_exp) * self.reward_norm
        dockscore = ((self.r_offset - normscore) * self.target_norm[1]) + self.target_norm[0]
        return dockscore

    def get_stats(self) -> Tuple[dict, List[dict]]:
        """ Return stats about new training molecules & new molecule infos to log (e.g. for topk) """
        # New molecules
        metrics = self._metrics

        new_mol_infos = []
        new_molecules = self.sampled_mols[self._last_ep_sampled_mol:]
        if len(new_molecules) > self._log_min_new:
            self._last_ep_sampled_mol = len(self.sampled_mols)
            new_mol_dockscore = [self.inv_r2r(x[0]) for x in new_molecules]
            metrics.update(["train_sampled_inv_r"], [new_mol_dockscore])

            # We should filter out here if there are any molecules that we do not want to send to
            # TopK - This will influence logging step of topK
            new_mol_infos = [x[-1] for x in new_molecules if len(x[-1]) > 0]

        log_metrics = self._metrics.get_stats(self._log_metrics_k)

        return log_metrics, new_mol_infos



