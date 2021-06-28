"""
Training a Classifier
"""
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import cv2
import copy

import torch
import numpy as np
import torch.optim as optim
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import wandb
from argparse import Namespace
from copy import deepcopy
import itertools
import sys
import time
import functools
import gym
from lightrl.policy.policy_base import Policy

from lightrl.env.vec_env import pre_process_obss
from lightrl.env.scores import ParallelSynth, QEDEstimator
from lightrl.env.gym_wrappers import OracleCandidateReward

from lightrl import env
from lightrl.utils.utils import set_seed
from lightrl.utils.utils import setup_loggers
from lightrl.reg_models import get_actor_model
from lightrl.env.vec_env import fast_from_data_list
from lightrl.utils.utils import SummaryStats
from lightrl.utils.radam_optimizer import RAdam
from lightrl.env.oracle import InterogateOracle

datasets_dir, programs_dir, summaries_dir = None, None, None



class StateActionReturnDataset(Dataset):

    def __init__(self,
                 data, block_size, actions, done_idxs, rtgs, timesteps, max_atoms,
                 max_steps=None, return_done=True):

        if block_size != 1:
            raise NotImplementedError

        self._max_steps = max_steps
        self._seq_block_size = block_size
        self.block_size = (max_atoms + 2) * block_size
        self.state_size = max_atoms
        self.return_done = return_done

        data, actions, done_idxs, rtgs, timesteps, av_idxs, horizon = self.process_dataset(
            data, actions, done_idxs, rtgs, timesteps
        )

        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.av_idxs = av_idxs
        self.horizon = horizon

    def process_dataset(self, data, actions, done_idxs, rtgs, timesteps):
        # Generate possible indxs - returning sequences of block_size len without done state in them
        horizon = np.zeros(len(actions))

        if self._max_steps is None or self._max_steps == 0:
            # Calculate horizon based on trajectory
            horizon_idxs = [-1] + done_idxs

            for i in range(1, len(horizon_idxs)):
                h_step = 0
                for p in range(horizon_idxs[i - 1] + 1, horizon_idxs[i] + 1)[::-1]:
                    horizon[p] = h_step
                    h_step += 1
        else:
            # calculate horizon based on max steps
            dones_bool = np.zeros(len(actions) + 1)
            dones_bool[np.array(done_idxs)] = 1
            step = self._max_steps
            for i in range(len(horizon)):
                horizon[i] = step
                step -= 1
                if dones_bool[i] == 1:
                    step = self._max_steps

        if self.return_done:
            av_idxs = np.where(horizon >= self._seq_block_size)[0]
        else:
            av_idxs = np.where(horizon >= self._seq_block_size-1)[0]

        return data, actions, done_idxs, rtgs, timesteps, av_idxs, horizon

    def __len__(self):
        return len(self.av_idxs)

    def __getitem__(self, idx):
        idx = self.av_idxs[idx]
        done_idx = idx + self._seq_block_size
        states = self.data[idx:done_idx] # (block_size, 4*84*84)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        # timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        timesteps = torch.tensor(self.horizon[idx:done_idx], dtype=torch.int64).unsqueeze(1)

        if self._max_steps == 0:
            raise NotImplementedError
            timesteps.zero_()

        _data = []
        for state, rtg, timestep, act in zip(states, rtgs, timesteps, actions):
            data = {
                "graph": state,
                "target": act.flatten(),
                "rcond": rtg.flatten(),
                "r_steps": timestep.flatten(),
            }
            _data.append(data)
        return _data[0]


class DebugDataset(StateActionReturnDataset):
    def __init__(self,
                 data, block_size, actions, done_idxs, rtgs, timesteps, max_atoms, traj_starts,
                 max_steps=None, return_done=True, max_traj=100, unique_start=True,
                 true_act=None):

        if block_size != 1:
            raise NotImplementedError

        self._max_steps = max_steps
        self._seq_block_size = block_size
        self.block_size = (max_atoms + 2) * block_size
        self.state_size = max_atoms
        self.num_states = block_size

        # ==========================================================================================
        # Debug custom
        timesteps = timesteps.tolist()

        # filter max trajectories
        self.start_end_idxs = start_end_idxs = []
        self.start_end = start_end = []
        self.start_traj = start_traj = []
        self.start_traj_rtg = start_traj_rtg = []
        start = data[0]
        start_smi = []
        pstart = 0
        selected_done = []
        filter_idx = []

        for i, pend in enumerate(done_idxs):
            if unique_start and traj_starts[i]["smiles"] in start_smi:
                pstart = pend + 1
                continue

            filter_idx += list(range(pstart, pend+1))
            selected_done.append(len(filter_idx)-1)

            start_smi.append(traj_starts[i]["smiles"])
            start_end_idxs.append([pstart, pend])
            start_end.append([data[pstart], data[pend]])
            traj_starts[i]["rtg"] = rtgs[pstart]
            traj_starts[i]["end_smiles"] = data[pend].smiles
            traj_starts[i]["actions"] = [true_act[iii] for iii in range(pstart, pend+1)]

            start_traj.append(traj_starts[i])
            start_traj_rtg.append(rtgs[pstart])
            start_smi.append(traj_starts[i]["smiles"])
            pstart = pend + 1
            if len(start_smi) > max_traj:
                break

        done_idxs = selected_done

        for lll in [actions, data, rtgs, timesteps]:
            lll[:] = [lll[x] for x in filter_idx]

        #f==========================================================================================
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

        # Generate possible indxs - returning sequences of block_size len without done state in them
        self.horizon = horizon = np.zeros(len(actions))

        if max_steps is None or max_steps == 0:
            # Calculate horizon based on trajectory
            horizon_idxs = [-1] + done_idxs

            for i in range(1, len(horizon_idxs)):
                h_step = 0
                for p in range(horizon_idxs[i - 1] + 1, horizon_idxs[i] + 1)[::-1]:
                    horizon[p] = h_step
                    h_step += 1
        else:
            # calculate horizon based on max steps
            dones_bool = np.zeros(len(actions) + 1)
            dones_bool[np.array(done_idxs)] = 1
            step = max_steps
            for i in range(len(horizon)):
                horizon[i] = step
                step -= 1
                if dones_bool[i] == 1:
                    step = max_steps

        if return_done:
            self.av_idxs = np.where(horizon >= block_size)[0]
        else:
            self.av_idxs = np.where(horizon >= block_size-1)[0]


class OnlineData(StateActionReturnDataset):
    def __init__(self, data_producer,
                 block_size, max_atoms, online_max_traj, online_start_traj,
                 max_steps=None, return_done=True,
                 true_act=None):

        self._data_producer = data_producer

        if block_size != 1:
            raise NotImplementedError

        self._max_steps = max_steps
        self._seq_block_size = block_size
        self.block_size = (max_atoms + 2) * block_size
        self.state_size = max_atoms
        self._min_horizon = block_size
        self.return_done = return_done
        self.online_max_traj = online_max_traj

        self.data = []
        self.actions = []
        self.done_idxs = []
        self.rtgs = []
        self.timesteps = []
        self.av_idxs = []
        self.horizon = []

        while len(self.done_idxs) < online_start_traj:
            time.sleep(30)
            self.collect_data()
            print(f"[OnlineData] We have so far {len(self.done_idxs)} "
                  f"(waiting for {online_start_traj})")

    def collect_data(self):
        data_producer = self._data_producer
        trajectories = data_producer.get_batch()
        print(f"[OnlineData] Got new data! ({len(trajectories)} num trajectories)")
        if len(trajectories) <= 0:
            return

        env_num_blocks = 105
        env_max_blocks = 7
        context_length = 1
        obss, actions, _, done_idxs, rtgs, timesteps, traj_start_mol, true_act = \
            build_condition_dataset(trajectories, max_blocks=env_max_blocks, num_blocks=env_num_blocks)

        data, actions, done_idxs, rtgs, timesteps, av_idxs, horizon = self.process_dataset(
            obss, actions, done_idxs, rtgs, timesteps
        )

        cnt_rem_dones = len(self.done_idxs) + len(done_idxs) - self.online_max_traj
        if cnt_rem_dones > 0 and len(self.done_idxs) > cnt_rem_dones:
            rem_dones = self.done_idxs[:cnt_rem_dones]
            new_start = rem_dones[-1] + 1
            # prev_rtg = [self.rtgs[x] for x in self.done_idxs][cnt_rem_dones:]

            self.done_idxs = [x - new_start for x in self.done_idxs[cnt_rem_dones:]]
            self.data = self.data[new_start:]
            self.actions = self.actions[new_start:]
            self.rtgs = self.rtgs[new_start:]
            self.timesteps = self.timesteps[new_start:]
            self.horizon = self.horizon[new_start:]

            # new_rtg = [self.rtgs[x] for x in self.done_idxs]

            # assert np.all(np.array(prev_rtg) == np.array(new_rtg)), "Bug in offset"
            # print("Cut from buffer")

        # must shift data
        done_idxs = [x + len(self.data) for x in done_idxs]

        self.data.extend(data)
        self.actions.extend(actions)
        self.done_idxs.extend(done_idxs)
        self.rtgs.extend(rtgs)
        self.timesteps.extend(timesteps.tolist())
        self.horizon.extend(horizon.tolist())

        horizons = np.array(self.horizon)
        if self.return_done:
            self.av_idxs = np.where(horizons >= self._seq_block_size)[0]
        else:
            self.av_idxs = np.where(horizons >= self._seq_block_size-1)[0]


def collate_fn(data, device):
    """ Used in dataloader to merge batch """
    _graphs = fast_from_data_list([x["graph"] for x in data], addk=["num_blocks"]).to(device)
    # _graphs["smiles"] = [x["smi"] for x in data]
    _tgt = torch.stack([x["target"] for x in data]).to(device)
    other = dict()
    for k in [x for x in data[0].keys() if x not in ["graph", "target"]]:
        # r_steps, rcond
        other[k] = torch.stack([x[k] for x in data]).to(device)

    return _graphs, _tgt, other


def train_epoch(ep, loader, model, optimizer, criterion, device, train_cfg):
    model.train()

    cond_steps = train_cfg.cond_steps

    correct = 0
    total = 0

    running_loss = 0.0

    eval_blocks = list(range(1, 9))
    eval_steps = list(range(8))
    acc_groups = dict({f"eval_step_{x}": [0, 0] for x in eval_steps})
    acc_groups.update(dict({f"eval_blocks_{x}": [0, 0] for x in eval_blocks}))

    for bidx, (data_graph, data_tgt, other) in enumerate(loader):
        r_steps, rcond = other["r_steps"], other["rcond"]
        if not cond_steps:
            r_steps_cond = torch.zeros_like(r_steps)
        else:
            r_steps_cond = r_steps

        data_graph, data_tgt = data_graph.to(device), data_tgt.to(device)
        data_tgt = data_tgt.flatten()
        x = Namespace(mol_graph=data_graph, rcond=rcond.flatten(), r_steps=r_steps_cond.flatten())
        optimizer.zero_grad()

        _, y_hat, _ = model(x, None, None)
        loss = criterion(y_hat, data_tgt)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(y_hat.data, 1)
        hits = (predicted == data_tgt)
        correct += hits.sum().item()

        # Calculate correct based on num blocks
        for x in eval_blocks:
            _select = (data_graph.num_blocks == x).flatten()
            _val = hits[_select]
            acc_groups[f"eval_blocks_{x}"][0] += hits[_select].sum().item()
            acc_groups[f"eval_blocks_{x}"][1] += len(_val)

        for x in eval_steps:
            _select = (r_steps == x).flatten()
            _val = hits[_select]
            acc_groups[f"eval_step_{x}"][0] += hits[_select].sum().item()
            acc_groups[f"eval_step_{x}"][1] += len(_val)

        total += data_tgt.size(0)

        if (bidx + 1) % 100 == 0:
            print(f"Done {bidx}")

    no_eval = []
    log_data = dict()
    for k, v in acc_groups.items():
        if v[1] > 0:
            log_data[f"{k}_f"] = v[0] / v[1]
            log_data[f"{k}_count"] = v[1]

    print(log_data)
    return correct / total, running_loss / total, log_data


def eval_epoch(loader, model, criterion, device, classification=True):
    model.eval()

    correct = 0
    total = 0

    topks = [1, 3, 5, 10, 20, 50]
    res = dict()
    for topk in topks:
        res[f"fmatch_top-{topk}"] = []
        res[f"top-{topk}"] = 0
        res[f"inv_top-{topk}"] = 0

    running_loss = 0.0
    with torch.no_grad():
        for bidx, (data_graph, data_tgt) in enumerate(loader):
            data_graph, data_tgt = data_graph.to(device), data_tgt.to(device)
            _, y_hat = model(data_graph)
            loss = criterion(y_hat, data_tgt)
            running_loss += loss.item()

            if classification:
                data_tgt_v = data_graph["target_v"]
            else:
                data_tgt_v = data_tgt_v

            # ======================================================================================
            # Calculate topk match
            tgt_sort = torch.argsort(data_tgt_v, axis=1, descending=True)
            pred_sort = torch.argsort(y_hat.data, axis=1, descending=True)

            # Go to bins groupping
            tgt_sort = data_graph.bins.gather(1, tgt_sort).data.cpu().numpy()
            pred_sort = data_graph.bins.gather(1, pred_sort).data.cpu().numpy()

            for topk in topks:
                top_target = tgt_sort[:, :topk]
                top_pred = pred_sort[:, :topk]
                for irow in range(len(tgt_sort)):
                    tgt_set = set(top_target[irow])
                    pred_set = set(top_pred[irow])
                    match = len(set.intersection(tgt_set, pred_set)) / len(set.union(tgt_set, pred_set))
                    res[f"fmatch_top-{topk}"].append(match)

                res[f"top-{topk}"] += ((top_pred - tgt_sort[:, 0][:, None]) == 0).any(axis=1).sum()
                res[f"inv_top-{topk}"] += ((top_target - pred_sort[:, 0][:, None]) == 0).any(axis=1).sum()

            if not classification:
                data_tgt = torch.argmax(data_tgt, dim=1)

            # ======================================================================================
            _, predicted = torch.max(y_hat.data, 1)
            total += data_tgt.size(0)
            correct += (predicted == data_tgt).sum().item()

    for topk in topks:
        res[f"fmatch_top-{topk}"] = np.mean(res[f"fmatch_top-{topk}"])
        res[f"top-{topk}"] = res[f"top-{topk}"] / total
        res[f"inv_top-{topk}"] = res[f"inv_top-{topk}"] / total

    return correct / total, running_loss / total, res


def build_condition_dataset(trajectories, max_blocks=7, num_blocks=105):
    """
    list of Tuple(
        list of (BlockMoleculeData, (block_idx, stem_idx)) - from start mol to end mo
        dockscore
        )
    """
    num_obs = sum([len(x[0]) for x in trajectories])
    obss = []
    actions = []
    rtgs = []
    done_idxs = []
    traj_start_mol = []
    true_act = []
    for traj, traj_mol_start, dockscore in trajectories:
        for mol, (block_idx, stem_idx), added_block_bond in traj:
            obss.append(mol)
            if block_idx is None:
                actions.append(0)
                true_act.append((None, None))
            else:
                actions.append(num_blocks * stem_idx + block_idx + max_blocks)
                true_act.append(added_block_bond)

        rtgs += [dockscore] * len(traj)  # 0 rtg on done
        done_idxs += [len(obss)-1]

        traj_start_mol.append(traj_mol_start)

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    return obss, actions, None, done_idxs, rtgs, timesteps, traj_start_mol, true_act


class EvalEnv:
    def __init__(self, model, device, max_steps=8, cond_steps=False, init_states=None):
        synth_net = ParallelSynth(use_cuda=not device.type == "cpu")
        synth_net.to(device)
        synth_net.share_memory()
        self.oracle = oracle = InterogateOracle()

        self.env = gym.make("BlockMolEnvGraph-v1", config={
            "obs_cuda": True,
            "random_steps": 1,
            "max_steps": max_steps,
            "allow_removal": False,
            "synth_net": synth_net,
            "oracle": oracle,
        }, proc_id=0)
        if init_states is None:
            self.env = OracleCandidateReward(self.env)

        self.cond_steps = cond_steps
        self.max_steps = max_steps
        self.device = device

        actor_critic = Policy(Namespace(), self.env.observation_space.shape, self.env.action_space,
                              model)
        actor_critic.to(device)

        self.actor_critic = actor_critic
        self._unique_smiles = set()
        self._debug = False
        self._debug_state_count = 0
        if not init_states:
            self._eval_configs = list(itertools.product(
                np.arange(-16, -14.0, 0.1), [True, False, False, False, False, False],
                [{"random_steps": x} for x in [1, 2, 3]]
            ))
        else:
            print([x["rtg"] for x in init_states])
            self._eval_configs = list(zip(
                [x["rtg"] for x in init_states], [True] * len(init_states), init_states
            ))
            self._debug_state_count = len(self._eval_configs)
            for i in range(10):
                self._eval_configs += list(zip(
                    [x["rtg"] for x in init_states], [False] * len(init_states), init_states
                ))

    def evaluate(self):
        eval_env = self.env

        actor_critic = self.actor_critic
        device = self.device
        actor_critic.eval()

        # list of [r_conditioning, deterministic]
        # eval_conf = list(itertools.product(np.arange(12, 12.2, 0.1), [True, False], [1, 2, 3]))
        eval_conf = self._eval_configs
        if self._debug_state_count > 0:
            debug_states = [0] * self._debug_state_count

        success = []
        debug_success = 0

        for icfg, (rtg, deterministic, init_cfg) in enumerate(eval_conf):
            random_steps = init_cfg.get("random_steps", -1)
            if random_steps != -1:
                debug = False
                eval_env.unwrapped.random_steps = random_steps
                obs = eval_env.reset()
            else:
                obs = eval_env.unwrapped._reset_load(init_cfg)
                debug = True

            rtg = np.around(rtg, 1)

            step_cnt = 0
            reward = 0
            done = False
            # print("Start", init_cfg)
            while not done:
                obs["rcond"] = rtg
                if not self.cond_steps:
                    obs["r_steps"] = 0

                p_obs = pre_process_obss([obs], device=device)
                with torch.no_grad():
                    res_m = actor_critic.act(p_obs, None, None, deterministic=deterministic)
                    action, eval_recurrent_hidden_states = res_m.action, res_m.rnn_hxs

                # Obser reward and next obs
                send_act = action.flatten().cpu().numpy().tolist()

                if self._debug:
                    if init_cfg["actions"][step_cnt][0] is None:
                        send_act[0] = 0
                    else:
                        mol = eval_env.unwrapped.molMDP.molecule
                        block_bonds = [[mol.blockidxs[x[0]], x[1]] for x in mol.stems]
                        stem_idx = block_bonds.index(list(init_cfg["actions"][step_cnt][:2]))
                        block_idx = init_cfg["actions"][step_cnt][2]
                        send_act[0] = 105 * stem_idx + block_idx + 7

                obs, reward, done, infos = eval_env.step(send_act[0])

                step_cnt += 1

            score_info = {
                "rcond": rtg,
                "deterministic": deterministic,
                "random_steps": random_steps
            }

            if not debug and reward == 1:
                success.append([score_info, infos["res_molecule"], None])
            elif debug:
                if send_act[0] == 0:
                    end_id = eval_env.unwrapped.molMDP.molecule.unique_id()
                    eval_env.unwrapped._reset_load(init_cfg["end_state"])
                    tgt_id = eval_env.unwrapped.molMDP.molecule.unique_id()

                    if end_id == tgt_id:
                        debug_states[icfg % self._debug_state_count] += 1
                        if icfg < self._debug_state_count:
                            debug_success += 1

        ret_req = self.oracle.get_req()

        if debug:
            print(f"[DEBUG] Reached: {debug_success/self._debug_state_count*100:.2f} |"
                  f"{np.sum(np.array(debug_states)>0)/self._debug_state_count*100:.2f} "
                  f"({len(eval_conf)})")

        # ==========================================================================================
        # Log a lot of stuff
        matches = 0
        close_matches = 0
        dockscores = []
        new_dockscores = []
        match_info = []
        for s in success:
            dockscore = s[2] = ret_req[s[1]]
            new_mol = False
            if s[1] not in self._unique_smiles:
                self._unique_smiles.update([s[1]])
                new_mol = True
            s[0]["new"] = True

            if dockscore is None:
                continue

            if new_mol:
                new_dockscores.append(dockscore)

            dockscores.append(dockscore)
            if dockscore == s[0]["rcond"]:
                matches += 1
                match_info.append(s[0])

            if np.abs(dockscore - s[0]["rcond"]) < 1.:
                close_matches += 1

        log_info = {
            "candidates_count": len(success),
            "candidates_f": len(success) / len(eval_conf),
            "match_count": matches,
            "match_f": matches / len(eval_conf),
            "close_match_count": close_matches,
            "close_match_f": close_matches / len(eval_conf),
        }
        if len(dockscores) > 0:
            log_info.update({
                "dockscore_min": np.min(dockscores),
                "dockscore_mean": np.mean(dockscores),
                "dockscore_cnt": len(dockscores),
            })
        else:
            log_info["dockscore_cnt"] = 0

        if len(new_dockscores) > 0:
            log_info.update({
                "new_dockscore_min": np.min(new_dockscores),
                "new_dockscore_mean": np.mean(new_dockscores),
                "new_dockscore_cnt": len(new_dockscores),
                "new_dockscore_f_candidates": len(new_dockscores) / len(success),
            })
        else:
            log_info.update({
                "new_dockscore_cnt": 0,
                "new_dockscore_f_suc": 0,
            })

        if len(match_info) > 0:
            match_det = np.array([x["deterministic"] for x in match_info]).astype(np.bool)
            match_rnd_steps = np.array([x["random_steps"] for x in match_info])

            log_info.update({
                "match_deterministic_cnt": np.sum(match_det),
                "match_nondeterministic_cnt": np.sum(~match_det),
            })

            for i in np.unique(match_rnd_steps):
                log_info[f"match_random_steps_{i}"] = np.sum(match_rnd_steps == i)

        return log_info, success


def run(cfg: Namespace):
    set_seed(cfg.seed, cfg.cuda)
    setup_loggers(cfg)
    do_plot = cfg.use_wandb
    out_dir = cfg.out_dir
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")
    summary_stats = SummaryStats(do_plot)
    max_steps = cfg.max_steps
    cond_steps = cfg.cond_steps

    # ==============================================================================================
    # Data
    if cfg.debug:
        cfg.online_dataset = False
    online_dataset = cfg.online_dataset
    context_length = 1
    online_dataset_collect_freq = cfg.online_dataset_collect_freq

    if online_dataset:
        from lightrl.utils.dataset_loaders import OnlineDataFeed
        data_feed = OnlineDataFeed(cfg.dataset_cfg)
        max_atoms = 70
        train_dataset = OnlineData(
            data_feed, context_length, max_atoms,
            cfg.dataset_cfg.online_max_traj, cfg.dataset_cfg.online_start_traj,
        )

    else:
        from lightrl.utils.dataset_loaders import load_predocked_dataset_with_children
        trajectories = load_predocked_dataset_with_children(cfg.dataset_cfg)
        env_num_blocks = 105
        env_max_blocks = 7
        obss, actions, _, done_idxs, rtgs, timesteps, traj_start_mol, true_act = \
            build_condition_dataset(trajectories, max_blocks=env_max_blocks, num_blocks=env_num_blocks)
        max_atoms = max([x.x.size(0) for x in obss])
        print(f"MAX ATOMS: {max_atoms}")

        if cfg.debug:
            train_dataset = DebugDataset(
                obss, context_length, actions, done_idxs, rtgs, timesteps, max_atoms, traj_start_mol,
                max_steps=None, max_traj=5000, unique_start=True,
                true_act=true_act
            )
        else:
            train_dataset = StateActionReturnDataset(
                obss, context_length, actions, done_idxs, rtgs, timesteps, max_atoms,
                max_steps=None
            )

    print(f"Training number of trajectories: {len(train_dataset.done_idxs)} "
          f"with {len(train_dataset)} transitions")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, collate_fn=functools.partial(collate_fn, device=device), shuffle=True)

    # Get data statistics

    # ==============================================================================================
    # Load model
    from lightrl.reg_models import get_actor_model
    from lightrl import reg_models
    from lightrl.models import get_model

    if cfg.model.name in reg_models.MODELS:
        model = get_actor_model(cfg.model)
    else:
        model = get_model(cfg.model, None, None)

    model = model.to(device)

    print(model)
    # ==============================================================================================
    init_states = None
    if cfg.debug:
        init_states = train_dataset.start_traj

    evaluator = EvalEnv(model, device, max_steps=max_steps, cond_steps=cond_steps,
                        init_states=init_states)

    # ==============================================================================================
    # Train
    criterion = nn.CrossEntropyLoss()
    optimizer = RAdam(model.parameters(), lr=cfg.lr)

    eval_freq = cfg.eval_freq
    log_freq = cfg.log_freq
    training_seen = 0
    best_train_acc = 0
    best_test_acc = 0
    for epoch in range(cfg.num_epochs):  # loop over the dataset multiple times
        acc, loss, other = train_epoch(
            epoch, train_loader, model, optimizer, criterion, device,
            Namespace(cond_steps=cond_steps)
        )

        if (epoch + 1) % log_freq == 0:
            print(f"[T] E: {epoch} | Acc: {acc*100:.3f}% | los: {loss:.5f}")

        training_seen += len(train_dataset)

        train_log = {"train_loss": loss, "train_acc": acc*100, "epoch": epoch, "training_seen": training_seen,
                       **other}
        summary_stats.update(train_log)

        wandb_log = copy.deepcopy(train_log)

        if (epoch + 1) % eval_freq == 0:
            eval_log, success = evaluator.evaluate()
            print(eval_log)
            print(success)
            summary_stats.update(eval_log)
            wandb_log.update(eval_log)

            """
            for name, _loader in eval_loaders:
                acc, loss, other = eval_epoch(
                    _loader, model, criterion, device, classification=classification
                )
                print(f"\n[E:{name}] E: {epoch} | Acc: {acc * 100:.3f}% | loss: {loss:.5f}\n")

                # Log eval
                eval_log = {
                        "loss": loss, "acc": acc * 100,
                        "epoch": epoch, "training_seen": training_seen,
                        **other
                    }

                for k in list(eval_log.keys()):
                    eval_log[f"{name}_{k}"] = eval_log.pop(k)

                summary_stats.update(eval_log)

                if do_plot:
                    wandb_log.update(eval_log)
            """

        # Log train
        if do_plot:
            wandb.log(wandb_log)

        if online_dataset and (epoch + 1) % online_dataset_collect_freq == 0:
            train_dataset.collect_data()
            print(f"Training number of trajectories: {len(train_dataset.done_idxs)} "
                  f"with {len(train_dataset)} transitions")

    print("FINISH it"); sys.stdout.flush(); time.sleep(5); os._exit(0)
    sys.exit()
    exit()
    quit()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn')
    except:
        print("already set context")

    from lightrl.utils.utils import parse_opts
    run(parse_opts())

