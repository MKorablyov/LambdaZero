from multiprocessing import Process, Pipe, Queue
import multiprocessing
import gym
from argparse import Namespace
from typing import List, Any
import numpy as np
import torch
import functools

from lightrl.env import gym_wrappers
from lightrl.env.dictlist import DictList
from torch_geometric.data import Batch



def _get_env(proc_id=0, seed=0, env_seed_offset=10000, env_name=None, env_args=None, env_wrapper=None):
    env_seed = seed + env_seed_offset * proc_id
    env = gym.make(env_name, config=env_args, proc_id=env_seed)
    env.unwrapped._env_proc_id = env_seed

    env = env_wrapper(env)
    env.seed(env_seed)
    return env


def get_envs(args: Namespace, no_envs: int, env_seed_offset: int = 10000, send_reset_info=False):
    # TODO should load envs on MP proc
    envs = []
    env_args = getattr(args, "env_args", None)
    actual_procs = getattr(args, "actual_procs", no_envs)

    env_args = dict() if env_args is None else vars(env_args)

    wrapper_methods = getattr(args, "wrapper", None)
    env_wrapper = get_wrappers(wrapper_methods)

    env_name = args.env

    # def _get_env(proc_id=0):
    #     env_seed = args.seed + env_seed_offset * proc_id
    #     env = gym.make(env_name, config=env_args, proc_id=env_seed)
    #     env.unwrapped._env_proc_id = env_seed
    #
    #     env = env_wrapper(env)
    #     env.seed(env_seed)
    #     return env

    envs.append([functools.partial(
        _get_env,
        proc_id=0, seed=args.seed, env_name=env_name, env_args=env_args, env_wrapper=env_wrapper
    )])

    chunk_size = int(np.ceil((no_envs - 1) / float(actual_procs)))
    if chunk_size > 0:
        for env_i in range(1, no_envs, chunk_size):
            env_chunk = []
            for i in range(env_i, min(env_i + chunk_size, no_envs)):
                env_chunk.append(functools.partial(
                    _get_env, # i
                    proc_id=i, seed=args.seed, env_name=env_name, env_args=env_args,
                    env_wrapper=env_wrapper
                ))
            envs.append(env_chunk)

    envs = ParallelEnv(envs, send_reset_info=send_reset_info)
    return envs, chunk_size


class RandStorage:
    def __init__(self, values):
        self.values = values

    def to(self, *args, **kwargs):
        pass

# This is mostly copied from torch_geometric, but with a few checks
# and generalities removed, as well as some speed improvements
def fast_from_data_list(data_list, inckeys=set(['stem_atmidx','edge_index','jbond_atmidx'])):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly.
    """

    keys = [x for x in data_list[0].keys if x != "smiles"]

    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}

    for key in keys:
        batch[key] = []

    cumsum = [0] * len(keys)
    batch.batch = []
    for j, key in enumerate(keys):
        cat_dim = 1 if key == 'edge_index' else 0
        slc = batch.__slices__[key]
        bk = batch[key]
        for i, data in enumerate(data_list):
            item = data[key]
            if cumsum[j] > 0:
                item = item + cumsum[j]
            slc.append(item.shape[cat_dim] + slc[-1])
            if key in inckeys:
                cumsum[j] += data.x.shape[0]
            bk.append(item)
            if j == 0:
                batch.batch.append(torch.full((data.x.shape[0], ), i, dtype=torch.long))

    for key in batch.keys:
        item = batch[key][0]
        cd = data_list[0].__cat_dim__(key, item)
        batch[key] = torch.cat(batch[key], dim=cd)

    ret_batch = batch.contiguous()
    if hasattr(data_list[0], "smiles"):
        ret_batch.smiles = [x.smiles for x in data_list]

    return ret_batch


def pre_process_obss(obss, device=None):
    obs = dict()

    for k in obss[0].keys():
        if k == "mol_graph":
            obs["mol_graph"] = fast_from_data_list([obs["mol_graph"] for obs in obss]).to(device)
        else:
            obs[k] = pre_process_vec([obs[k] for obs in obss], device=device)

    return DictList(obs)


def pre_process_vec(vec, device=None):
    if isinstance(vec[0], torch.Tensor):
        return torch.stack(vec).to(device)

    vec = np.array(vec)
    vec = torch.tensor(vec, device=device)  # type: torch.Tensor
    return vec


def idem(x):
    return x


def env_wrapp(w_env, env_wrappers):
    for wrapper in env_wrappers[::-1]:
        w_env = wrapper(w_env)
    return w_env


def get_wrappers(wrappers: List[str]):
    # Get env wrappers - must be a list of elements
    if wrappers is None:

        env_wrapper = idem
    else:
        env_wrappers = [getattr(gym_wrappers, w_p) for w_p in wrappers]
        env_wrapper = functools.partial(env_wrapp, env_wrappers=env_wrappers)
    return env_wrapper


def worker_multi(conn, conn_send, envs):
    envs = list(envs)

    while True:
        cmd, datas = conn.recv()
        if cmd == "step":
            for (env_idx, env), data in zip(envs, datas):
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn_send.put((env_idx, (obs, reward, done, info)))
        elif cmd == "reset":
            for env_idx, env in envs:
                obs = env.reset()
                conn_send.put((env_idx, obs))
        elif cmd == "exit":
            return
        else:
            raise NotImplementedError


def worker_multi_advanced(conn, conn_send, conn_reset, envs_make):
    envs = [(env_idx, _make()) for env_idx, _make  in envs_make]

    if conn_reset is None:
        def _reset(env):
            obs = env.reset()
            return True, obs
    else:
        def _reset(env):
            cmd, datas = conn_reset.get()
            if cmd is None:
                return False, None
            obs = env.reset(**datas)
            return True, obs

    while True:
        cmd, datas = conn.recv()
        if cmd == "step":
            for (env_idx, env), data in zip(envs, datas):
                obs, reward, done, info = env.step(data)
                # info["last_obs"] = obs
                if done:
                    ok, obs = _reset(env)
                    if not ok:
                        continue

                conn_send.put((env_idx, (obs, reward, done, info)))
        elif cmd == "reset":
            for env_idx, env in envs:
                ok, obs = _reset(env)
                if not ok:
                    continue

                conn_send.put((env_idx, obs))
        elif cmd == "exit":
            return
        else:
            raise NotImplementedError


def worker_multi_with_last_obs(conn, conn_send, envs):
    envs = list(envs)

    while True:
        cmd, datas = conn.recv()
        if cmd == "step":
            for (env_idx, env), data in zip(envs, datas):
                obs, reward, done, info = env.step(data)
                if done:
                    if info is None:
                        info = dict()
                    info["last_obs"] = obs
                    obs = env.reset()
                conn_send.put((env_idx, (obs, reward, done, info)))
        elif cmd == "reset":
            for env_idx, env in envs:
                obs = env.reset()
                conn_send.put((env_idx, obs))
        elif cmd == "exit":
            return
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """
        A concurrent execution of environments in multiple processes.
        Distribute envs to limited number of processes
    """

    def __init__(self, envs, worker_target=worker_multi_advanced, send_reset_info=False):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.first_env = self.envs[0][0]()  # Move first env to thread
        self.observation_space = self.first_env.observation_space
        self.action_space = self.first_env.action_space

        self.locals = []

        self.no_envs = sum(map(len, self.envs[1:]))
        self.num_procs = self.no_envs + 1
        self.local_recv = remote_send = Queue()
        self.send_reset_info = send_reset_info
        send_reset_q = None
        if send_reset_info:
            self.send_reset_q = send_reset_q = Queue()

        env_idx = 1
        self.env_idxs = []
        self.processes = []
        for env_b in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(
                target=worker_target,
                args=(remote, remote_send, send_reset_q,
                      zip(range(env_idx, env_idx+len(env_b)), env_b)))
            p.daemon = True
            p.start()
            remote.close()
            self.env_idxs.append([env_idx, env_idx+len(env_b)])
            env_idx += len(env_b)
            self.processes.append(p)

    def reset(self, num_evals: int = 0):
        if self.send_reset_info:
            # Empty que
            while not self.send_reset_q.empty():
                self.send_reset_q.get()

            # Must fill the que of reset infos (add safe guard of 2xnum_procs
            for i in range(num_evals + self.num_procs * 2):
                self.send_reset_q.put(("info", dict(episode=i)))

        for local in self.locals:
            local.send(("reset", None))

        if self.send_reset_info:
            _, r_info = self.send_reset_q.get()
            results = [self.first_env.reset(**r_info)] + [None] * self.no_envs
        else:
            results = [self.first_env.reset()] + [None] * self.no_envs

        no_recv = 0
        max_recv = self.no_envs

        local = self.local_recv
        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1
        return results

    def step(self, actions):
        # Send Chunck actions
        for local, action_idxs in zip(self.locals, self.env_idxs):
            local.send(("step", actions[action_idxs[0]:action_idxs[1]]))
        obs, reward, done, info = self.first_env.step(actions[0])
        if done:
            # info["last_obs"] = info
            if self.send_reset_info:
                _, r_info = self.send_reset_q.get()
                obs = self.first_env.reset(**r_info)
            else:
                obs = self.first_env.reset()

        results = [(obs, reward, done, info)] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1

        results = zip(*results)
        return results

    def render(self):
        raise NotImplementedError

    def close_procs(self):
        for local in self.locals:
            local.send(("exit", None))

        for p in self.processes:
            p.terminate()
            p.join()


class ParallelEnvWithLastObs(ParallelEnv):
    """
        Send last observation before reset.
    """

    def __init__(self, *args, **kwargs):
        multiprocessing.connection.BUFSIZE *= 10
        super().__init__(*args, worker_target=worker_multi_with_last_obs, **kwargs)

    def step(self, actions):
        # Send Chunck actions
        for local, action_idxs in zip(self.locals, self.env_idxs):
            local.send(("step", actions[action_idxs[0]:action_idxs[1]]))
        obs, reward, done, info = self.first_env.step(actions[0])
        if done:
            if info is None:
                info = dict()
            info["last_obs"] = obs
            obs = self.first_env.reset()

        results = [(obs, reward, done, info)] + [None] * self.no_envs
        no_recv = 0
        max_recv = self.no_envs
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1

        results = zip(*results)
        return results


class ParallelEnvWithLastObsIndex(ParallelEnv):
    """
        Send last observation before reset.
    """

    def __init__(self, envs, **kwargs):
        all_envs = []
        for x in envs:
            all_envs += x
        all_envs = [[x] for x in all_envs]
        super().__init__(all_envs, worker_target=worker_multi_with_last_obs, **kwargs)

    def step(self, actions, idxs=None):
        if idxs is None:
            idxs = list(range(self.no_envs + 1))

        # Send Chunck actions
        locals = self.locals
        send_to_first = False
        results = [None] * (self.no_envs + 1)
        max_recv = len(idxs)

        for send_act, send_idx in zip(actions, idxs):
            if send_idx != 0:
                locals[send_idx-1].send(("step", [send_act]))
            else:
                send_to_first = True

        if send_to_first:
            obs, reward, done, info = self.first_env.step(actions[0])

            if done:
                if info is None:
                    info = dict()
                info["last_obs"] = obs
                obs = self.first_env.reset()

            results[0] = (obs, reward, done, info)
            max_recv -= 1

        no_recv = 0
        local = self.local_recv

        while no_recv < max_recv:
            env_idx, r = local.get()
            results[env_idx] = r
            no_recv += 1

        results = zip(*[x for x in results if x is not None])
        return results
