import cv2
import sys
import gym
import json
import pandas as pd
import numpy as np
from multiprocessing import Pool
from argparse import Namespace
import itertools
import time
import torch
import copy
from tqdm import tqdm
from multiprocessing import Process, Pipe, Queue
import random
import _pickle as cPickle

from LambdaZero.contrib.oracle.oracle_predocked import PreDockingDB
from LambdaZero.examples.lightrl import env
from LambdaZero.examples.lightrl.env.reward import DummyReward
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def service_get_mol_graph(recv, send):
    env = gym.make("BlockMolEnvGraph-v1", config={
        "obs_cuda": False,
    }, proc_id=0)
    transform = None
    env.molMDP.build_translation_table(fill_in=False)

    while True:
        cmd, states = recv.recv()
        if cmd == 1:
            args = zip(states,
                       itertools.repeat(env.molMDP),
                       itertools.repeat(env.graph_mol_obs),
                       itertools.repeat(transform))
            for arg in args:
                ret = pool_get_mol_graph(arg)
                # send.put(ret)
                send.put(cPickle.dumps(ret, -1))
        else:
            return


def service_sample_mol_graph(recv, send):
    env = gym.make("BlockMolEnvGraph-v1", config={
        "obs_cuda": False,
    }, proc_id=0)

    rnd_state = np.random.RandomState(13)

    # Filter based on max_branches & num_blocks
    max_branches = env.unwrapped.max_branches
    num_blocks = env.unwrapped.num_blocks

    transform = None
    env.molMDP.build_translation_table(fill_in=False)

    all_states = []
    bins = []
    sample_weights = []

    while True:
        cmd, cmd_info = recv.recv()
        if cmd == 1:
            all_states += cmd_info
            bins += [x["bin"] for x in cmd_info]
            _nbins = np.array(bins)
            sample_weights = 1/np.bincount(_nbins)[_nbins]
            ids = [x["id"] for x in all_states]
            # print(f"LOADED IDS: [{min(ids)} - {max(ids)}] unique: {len(np.unique(ids))}")
            # print(f"Histogram sample_weights: {np.histogram(sample_weights, bins=10)}")
        elif cmd == 2:
            b_size = cmd_info
            states = random.choices(population=all_states, weights=sample_weights, k=b_size)

            # Generate new ids
            for x in states:
                x["id"] += 1

            args = zip(states,
                       itertools.repeat(env.molMDP),
                       itertools.repeat(env.graph_mol_obs),
                       itertools.repeat(transform))

            for arg in args:
                ret = pool_get_mol_graph(arg, max_branches=max_branches, num_blocks=num_blocks)
                send.put(cPickle.dumps(ret, -1))
        else:
            return


def pool_get_mol_graph(largs, max_branches=20, num_blocks=105):
    state, env_molMDP, env_graph_mol_obs, transform = largs
    _rnd = np.random.RandomState(state["id"])

    if np.any(np.array(state["blockidxs"]) >= num_blocks):
        return (None, None, None)

    mol = env_molMDP.load(state)
    graph, flat_graph = env_graph_mol_obs(mol, flatten=False)
    if transform is not None:
        graph = transform(graph)

    error = False
    try:
        graph.smiles = mol.smiles
    except:
        error = True

    graph.num_blocks = torch.tensor(len(mol.blockidxs))

    traj = []
    traj_mol = mol
    # print("START", state)

    while True:
        parents = env_molMDP.parents(traj_mol)
        if len(parents) == 0:
            error = True
            break
        parent_idx = _rnd.randint(len(parents))
        parent = list(parents[parent_idx])
        blockidx, stemidx = parent[1]
        if stemidx >= max_branches:
            error = True
            break

        traj.append(parent)
        traj_mol = parent[0]
        numblocks = traj_mol.numblocks
        parent[0] = env_graph_mol_obs(traj_mol, flatten=False)[0]
        parent[0].num_blocks = torch.tensor(numblocks)

        try:
            mol_smile = traj_mol.smiles
        except:
            error = True

        if error:
            break

        if numblocks <= 1:
            break

    if error:
        return (None, None, None)

    start_state = traj_mol.dump()
    start_state["end_state"] = state

    traj = traj[::-1]
    traj.append([graph, (None, None), None])

    ret_data = (traj, start_state, state["dockscore"])

    return ret_data


def load_predocked_df(args):
    # Get config
    sample_size = getattr(args, "sample_size", 0)
    filter_by_score = getattr(args, "filter_candidates", False)
    with_scores = getattr(args, "with_scores", False) or filter_by_score
    qed_th = getattr(args, "qed_th", 0.3)
    synth_th = getattr(args, "synth_th", 4.)
    dockscore_th = getattr(args, "dockscore_th", None)
    dockscore_unique = getattr(args, "dockscore_unique", False)
    dockscore_uniform_sample = getattr(args, "dockscore_uniform_sample", False)
    filter_smiles = getattr(args, "filter_smiles", None)
    random_state = getattr(args, "random_state", 13)

    gather_sample_size = sample_size
    if filter_by_score:
        gather_sample_size = 0

    # Get predocked db
    df = PreDockingDB.read_last_db(sample_size=gather_sample_size, with_scores=with_scores)
    orig_set_len = len(df)

    if dockscore_unique:
        df = df[~df.index.duplicated()]

    if filter_smiles is not None:
        filter_by_smi = df.index.isin(filter_smiles)
        print(f"Filtering out {filter_by_smi.sum()} rows with ignored smiles!")
        df = df[~filter_by_smi]

    if filter_by_score:
        df = df[(df["qed_score"] >= qed_th) & (df["synth_score"] >= synth_th)]

        if dockscore_th is not None:
            df = df[df["dockscore"] <= dockscore_th]

        if sample_size != 0:
            if dockscore_uniform_sample:
                hist_v = np.histogram(df.dockscore.values, bins=10)
                bins = np.digitize(df.dockscore.values, hist_v[1])
                sample_weights = 1 / np.bincount(bins)[bins]
                df = df.sample(sample_size, weights=sample_weights, random_state=random_state)
            else:
                df = df.iloc[:sample_size]

    # Print some stats
    hist_v = np.histogram(df.dockscore.values, bins=100)
    dock_hist = list(zip(*hist_v))
    print("Dock 100 histogram\n", dock_hist)
    bins = np.digitize(df.dockscore.values, hist_v[1])
    df["bin"] = bins

    print(f"Loaded after filtering {len(df)} (/{orig_set_len}) states with {len(np.unique(df.index))} unique smiles")

    df["id"] = np.arange(len(df))
    return df


class OnlineDataFeed:
    def __init__(self, args: Namespace):
        num_pool = getattr(args, "mol_graph_pool", 10)
        self._req_batch_size = getattr(args, "online_batch_size", 4000) // num_pool
        self._req_buffer = getattr(args, "req_buffer", 2)

        df = load_predocked_df(args)
        state_records = df.to_dict(orient='records')
        print(f"Starting pool with {num_pool} children for {len(state_records)} records")

        self._recv = recv = Queue()

        bsize = int(np.ceil(len(state_records) / num_pool))
        self.processes = processes = []
        self.pipes = pipes = []
        for i in range(num_pool):
            local, remote = Pipe()
            pipes.append(local)
            p = Process(
                target=service_sample_mol_graph,
                args=(remote, recv))
            p.start()

            local.send((1, state_records[i * bsize: (i + 1) * bsize]))
            processes.append(p)

        self._traj_que = []
        self._received_count = 0
        self._last_req_count = 0
        self._requested_counts = []

        self._send_req()

    def _send_req(self):
        while len(self._requested_counts) < self._req_buffer:
            for local in self.pipes:
                local.send((2, self._req_batch_size))

            self._last_req_count += self.request_size
            self._requested_counts.append(self._last_req_count)

    @property
    def request_size(self):
        return self._req_batch_size * len(self.pipes)

    def _receive_data(self):
        recv = self._recv

        recv_cnt = 0
        all_graph_traj = []

        while not recv.empty():
            res = recv.get()
            res = cPickle.loads(res)
            if res[0] is not None:
                all_graph_traj.append(res)
            recv_cnt += 1

        self._traj_que += all_graph_traj
        self._received_count += recv_cnt

    def get_batch(self):
        # Receive data in que
        self._receive_data()

        # Check if enough received msgs (based on batch request size
        if self._received_count < self._requested_counts[0]:
            return []

        # Consider solved the first request in que and pop - in order to get a new batch
        self._requested_counts.pop(0)

        # Send request until buffer max size
        self._send_req()

        ret_traj = self._traj_que
        self._traj_que = list()
        return ret_traj

    def close(self):
        for send in self.pipes:
            send.send((0, None))

        for p in self.processes:
            p.terminate()
            p.join()


def load_predocked_dataset_with_children(args: Namespace):
    num_pool = getattr(args, "mol_graph_pool", 10)
    duplicate_states = getattr(args, "duplicate_states", 1)

    df = load_predocked_df(args)
    df.reset_index(inplace=True)
    state_records = df.to_dict(orient='records')

    # Generate duplicates but with different ids for sampling different trajectories
    if duplicate_states > 1:
        all_states = list(state_records)
        for idp in range(1, duplicate_states):
            new_states = copy.deepcopy(state_records)
            for x in new_states:
                x["id"] += idp
            all_states += new_states
        state_records = all_states

    print(f"Starting pool with {num_pool} children for {len(state_records)} records")

    st = time.time()

    recv = Queue()

    bsize = int(np.ceil(len(state_records) / num_pool))
    processes = []
    pipes = []
    pipes_remote = []
    for i in range(num_pool):
        local, remote = Pipe()
        pipes.append(local)
        pipes_remote.append(remote)
        p = Process(
            target=service_get_mol_graph,
            args=(remote, recv))
        p.start()

        local.send((1, state_records[i*bsize: (i+1)*bsize]))
        processes.append(p)

    recv_cnt = 0
    all_graph_traj = []
    while recv_cnt < len(state_records):
        res = recv.get()
        res = cPickle.loads(res)
        if res[0] is not None:
            all_graph_traj.append(res)
        recv_cnt += 1

        if recv_cnt % 1000 == 0:
            done_time = time.time() - st
            estimated_remaining = (done_time / recv_cnt) * (len(state_records) - recv_cnt)
            print(f"Done {recv_cnt}/{len(state_records)} in {done_time:.2f} s"
                  f" (~remaining {estimated_remaining:.2f} s)")

    print(f"FINAL DONE {len(state_records)} resulted in {len(all_graph_traj)} trajectories.")
    all_graph_traj = [x for x in all_graph_traj if x[0] is not None]

    for send in pipes:
        send.send((0, None))

    for ppp in pipes:
        ppp.close()

    for ppp in pipes_remote:
        ppp.close()

    for p in processes:
        p.terminate()
        p.join()

    """"
    # Load Env to have methods for loading molMDP from state and converting them to graphs
    env = gym.make("BlockMolEnvGraph-v1", config={
        "obs_cuda": False,
    }, proc_id=0)
    transform = None
    env.molMDP.build_translation_table(fill_in=False)

    with torch.multiprocessing.Pool(num_pool) as p:
        bsize = 1000
        batch_idxs = range(0, len(state_records), bsize)

        for ibtch, batch in tqdm(enumerate(batch_idxs), total=len(batch_idxs)):
            graphs_with_traj = p.map(
                pool_get_mol_graph,
                zip(state_records[batch: batch + bsize],
                    itertools.repeat(env.molMDP),
                    itertools.repeat(env.graph_mol_obs),
                    itertools.repeat(transform))
            )

            # Free mem from shared
            clone_g = [copy.deepcopy(x) for x in graphs_with_traj]
            for x in graphs_with_traj:
                del x
            del graphs_with_traj
            all_graph_traj += [x for x in clone_g if x[0] is not None]
    """

    """
        # Serial solution In case of problems with parallel version
    for ibtch in range(len(state_records)):
        all_graph_traj.append(
            pool_get_mol_graph((state_records[ibtch], env.molMDP, env.graph_mol_obs, transform))
        )
        if (ibtch + 1) % 1000 == 0:
            print(f"DONE : {len(all_graph_traj)}")
    """

    print(f"Finish loading in {time.time() - st}")

    # for iii, (traj, dockscore, traj_mol) in enumerate(trajectories):
    #     env.reset()
    #     env.molMDP.reset()
    #     env.molMDP.molecule = traj_mol
    #     env.num_steps = 0
    #
    #     _obs = env._make_obs()[0]
    #     print(
    #         traj_mol.blockidxs,
    #         traj_mol.slices,
    #         traj_mol.jbonds,
    #         traj_mol.stems,
    #     )
    #     print(
    #         df.to_dict(orient='records')[iii]
    #     )
    #     # print((_obs["mol_graph"].x.cpu() == traj[0][0].x).all(), (_obs["mol_graph"].edge_attr.cpu() == traj[0][0].edge_attr).all())
    #     for itr, (mol, (block_idx, stem_idx)) in enumerate(traj[:-1]):
    #         print(itr, block_idx, stem_idx)
    #         obs, _, _, _  = env.step(105 * stem_idx + block_idx + 7)
    #         x, y = obs["mol_graph"].x.shape == traj[itr+1][0].x.shape, obs["mol_graph"].edge_attr.shape == traj[itr+1][0].edge_attr.shape
    #         print(itr, x, y)
    #         assert x and y, "Not the ssame"
    # import pdb;
    # pdb.set_trace()

    return all_graph_traj


if __name__ == "__main__":
    args = Namespace(sample_size=0, filter_candidates=True)
    traj = load_predocked_dataset_with_children(args)
    print(len(traj))