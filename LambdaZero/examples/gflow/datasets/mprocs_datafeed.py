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
import _pickle as cPickle

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def get_sample_model(send_que: Queue, args, model, proxy):
    sample_proc_buffer_size = args.sample_proc_buffer_size
    dataset = DataGenerator(args, args.device, model=model, proxy=proxy, sample_prob=args.sample_prob)

    while True:
        sample = dataset.get_sample_model()
        # send_que.put(sample)
        send_que.put(cPickle.dumps(sample, -1))
        # print("SAMPLE QUE size", send_que.qsize())
        while send_que.qsize() > sample_proc_buffer_size:
            pass


def get_batches(send_que: Queue, recv_samples_que: Queue, args):
    _id = np.random.randint(999)

    sample_batch_buffer_size = args.sample_batch_buffer_size
    dataset = DataGenerator(args, args.device)

    batch_size = 16
    batches = []
    while True:
        samples = []
        for i in range(batch_size):
            sample, (r, m, trajectory_stats, inflow, info) = recv_samples_que.get()
            samples += sample

        batches.append(dataset.sample2batch(samples))

        send_que.put(batches.pop(0))

        while send_que.qsize() > sample_batch_buffer_size:
            pass


def transform_mol(recv: Queue, send: Queue, args: Namespace):
    bpath = osp.join(datasets_dir, getattr(args.mdp_init, "bpath", "fragdb/blocks_PDB_105.json"))
    mdp = MolMDPExtended(bpath)  # Used for generating representation
    mdp_init = getattr(args, "mdp_init", {"repr_type": "atom_graph"})
    mdp_init = getattr(mdp_init, "__dict__", mdp_init)
    mdp_init["device"] = torch.device("cpu")
    mdp.post_init(**mdp_init)

    while True:
        _id, mol = recv.get()
        repr = mdp.mol2repr(cPickle.loads(mol))
        send.put((_id, cPickle.dumps(repr, -1)))


class OnlineDataFeed(DataGenerator):
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        super(OnlineDataFeed, self).__init__(args, device, model, proxy, sample_prob)

        # Variables for online DataFeed
        self._sample_processes = []
        self._recv_sample = None
        self._btime = deque(maxlen=100)
        self._btime1 = deque(maxlen=100)
        self._btime2 = deque(maxlen=100)
        self._btime3 = deque(maxlen=100)
        self._btime4 = deque(maxlen=100)

    def _sample_train_transitions(self, batch_size: int):
        # sample batch of transitions
        samples = random.choices(self.train_transition_buffer, k=batch_size)
        return samples

    def _init_parallel_sampling(self):
        args = self.args

        assert len(self._sample_processes) == 0, "Already initialized parallel sampling"

        model = self.sampling_model
        proxy = self.proxy_reward
        model.share_memory()
        proxy.share_memory()

        # Processes for sampling new trajectories using Model
        sample_procs = getattr(args, "sample_procs", 4)
        self._recv_sample = recv = Queue()
        for i in range(sample_procs):
            p = Process(
                target=get_sample_model,
                args=(recv, args, model, proxy))
            p.start()
            self._sample_processes.append(p)

        # Processes for batching data - Should work with self.sample_batch_procs
        batch_procs = getattr(args, "batch_procs", 4)
        self._batch_buffer_proc = []
        self._recv_batch = recv = Queue()
        for i in range(batch_procs):
            p = Process(
                target=get_batches,
                args=(recv, self._recv_sample, args))
            p.start()
            self._batch_buffer_proc.append(p)

        # Processes for converting BlockMoleculeDataExtended to graph representation
        repr_procs = getattr(args, "repr_procs", 10)

        self._repr_procs = []
        self._repr_send = send_q = Queue()
        self._repr_recv = recv_q = Queue()
        for i in range(repr_procs):
            p = Process(
                target=transform_mol,
                args=(send_q, recv_q, args))
            p.start()
            self._repr_procs.append(p)

    def set_sampling_model(self,
                           model: torch.nn.Module = None, proxy: BaseProxy = None,
                           sample_prob: float = 0.5):

        super().set_sampling_model(model, proxy, sample_prob)

        self._init_parallel_sampling()

    def get_sample_model(self):
        recv_sample = self._recv_sample.get()
        sample, (r, m, trajectory_stats, inflow, info) = cPickle.loads(recv_sample)
        return sample, (r, m, trajectory_stats, inflow, info)

    # def _sample_new_trajectory(self):
    #     recv_sample = self._recv_sample.get()
    #     sample, (r, m, trajectory_stats, inflow, info) = cPickle.loads(recv_sample)
    #
    #     self.sampled_mols.append((r, m, trajectory_stats, inflow, info))
    #
    #     # Add training Transition
    #     self.train_transition_buffer.extend(sample)
    #     print(len(self.train_transition_buffer))
    #
    #     # should have here method to cut train_transitions
    #     max_buffer = self.args.max_transition_buffer_size
    #     if len(self.train_transition_buffer) > max_buffer:
    #         self.train_transition_buffer = self.train_transition_buffer[-max_buffer:]
    #
    #     return sample, (r, m, trajectory_stats, inflow, info)
    #
    # def sample_train_batch(self, batch_size: int):
    #     """
    #         [E 403] [TRAIN] GFlow start train
    #         BATCHING TIME1:  0.000646049976348877
    #         BATCHING TIME2:  0.0812917947769165
    #         BATCHING TIME3:  0.03232088804244995
    #         BATCHING TIME4:  0.002609410285949707
    #         BATCHING TIME:  0.11763051271438599
    #         sample: 0.254079
    #         train: 0.116315
    #         inflow: 0.023552
    #         outflow: 0.001831
    #         lossbackward: 0.064788
    #         optstep: 0.025931
    #     """
    #
    #     # Sample new trajectories
    #     for i in range(self.args.iter_sample_new_traj):
    #         self._sample_new_trajectory()
    #
    #     # sample batch of transitions
    #     samples = self.train_rng.choice(self.train_transition_buffer, batch_size)
    #
    #     # return samples
    #     batch = self.sample2batch(samples)
    #     return batch

    def sample_batch_procs(self, batch_size: int):
        batch = self._recv_batch.get()
        return batch

    def sample2batch(self, samples: List[Any]):
        st = time.time()

        p, a, r, s, d = list(zip(*samples))

        mols = (p, s)
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []),
                               device=self._device).long()
        self._btime1.append(time.time() - st)
        st = time.time()

        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch

        # # Send to multiprocs
        mol_p = sum(p, ())
        mol_s = s

        _send_batch = list(mol_p) + list(mol_s)
        for _id, _mol in enumerate(_send_batch):
            self._repr_send.put((_id, cPickle.dumps(_mol, -1)))  # Pickle improves a lot here!

        recv_count = 0
        while recv_count < len(_send_batch):
            _id, _molr = self._repr_recv.get()
            _send_batch[_id] = cPickle.loads(_molr)  # Pickle improves a lot here!
            recv_count += 1

        mol_p = _send_batch[:len(mol_p)]
        mol_s = _send_batch[len(mol_p):]
        self._btime2.append(time.time() - st)
        st = time.time()

        p = self.mdp.mols2batch(mol_p)
        s = self.mdp.mols2batch(mol_s)
        self._btime3.append(time.time() - st)
        st = time.time()

        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)
        self._btime4.append(time.time() - st)

        # print("BATCHING TIME1: ", np.mean(self._btime1))
        # print("BATCHING TIME2: ", np.mean(self._btime2))
        # print("BATCHING TIME3: ", np.mean(self._btime3))
        # print("BATCHING TIME4: ", np.mean(self._btime4))

        return (p, p_batch, a, r, s, d, mols)

    def close(self):
        for p in self._sample_processes:
            p.terminate()
            p.join()
        for p in self._batch_buffer_proc:
            p.terminate()
            p.join()
        for p in self._repr_procs:
            p.terminate()
            p.join()
