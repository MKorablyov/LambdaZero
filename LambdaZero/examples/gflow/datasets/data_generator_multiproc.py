from argparse import Namespace
import numpy as np
import torch
from torch.multiprocessing import Process, Queue, Pipe
import os.path as osp
import random
import pickle
from typing import List, Any, Tuple


from LambdaZero.examples.gflow.datasets.data_generator import DataGenerator, TrainBatch, Transition
from LambdaZero.examples.proxys.base_proxy import BaseProxy
from LambdaZero.utils import get_external_dirs
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def get_sample_model(send_que: Pipe, recv_que: Pipe, args, model, proxy):
    dataset = DataGenerator(
        args, torch.device("cpu"), model=model, proxy=proxy, sample_prob=args.sample_prob
    )

    def sample_model(mols: List[BlockMoleculeDataExtended], no_stop=False):
        batch = [dataset.mdp.mol2repr(x) for x in mols]
        send_que.send(pickle.dumps((1, (batch, no_stop)), -1))
        req = recv_que.recv()
        # ret = cPickle.loads(req)
        return req

    def get_reward(m: BlockMoleculeDataExtended):
        return -1., {}

    dataset._sample_model = sample_model
    dataset._get_reward = get_reward

    with torch.no_grad():
        while True:
            # request sample
            sample = dataset.get_sample_model()

            send_que.send(pickle.dumps((0, sample), -1))


def transform_mol(recv: Queue, send: Queue, args: Namespace):
    bpath = osp.join(datasets_dir, getattr(args.mdp_init, "bpath", "fragdb/blocks_PDB_105.json"))
    mdp = MolMDPExtended(bpath)  # Used for generating representation
    mdp_init = getattr(args, "mdp_init", {"repr_type": "atom_graph"})
    mdp_init = getattr(mdp_init, "__dict__", mdp_init)
    mdp_init["device"] = torch.device("cpu")
    mdp.post_init(**mdp_init)

    while True:
        _id, mol = recv.get()
        repr = mdp.mol2repr(pickle.loads(mol))
        send.put((_id, pickle.dumps(repr, -1)))


class DataGeneratorMultiProc(DataGenerator):
    def __init__(self, args: Namespace, device: torch.device,
                 model: torch.nn.Module = None, proxy: BaseProxy = None, sample_prob: float = 0.5):

        super(DataGeneratorMultiProc, self).__init__(args, device, model, proxy, sample_prob)

        # Variables for online DataFeed
        self._sample_processes = []
        self._recv_sample = None

    def _init_parallel_sampling(self):
        args = self.args

        assert len(self._sample_processes) == 0, "Already initialized parallel sampling"

        proxy = None

        # Processes for sampling new trajectories using Model
        sample_procs = min(getattr(args, "sample_procs", 4), self._iter_sample_new_traj)
        self._recv_sample = []
        self._send_sample = []

        for i in range(sample_procs):
            local1, remote1 = Pipe()
            local2, remote2 = Pipe()
            self._recv_sample.append(local1)
            self._send_sample.append(local2)
            p = Process(
                target=get_sample_model,
                args=(remote1, remote2, args, None, proxy))
            p.start()
            self._sample_processes.append(p)

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

    def _sample_batch_trajectories(self, num_traj):
        """ Just parallelize with multiple procs """
        recv_sample = self._recv_sample
        send_sample = self._send_sample

        # Sample new trajectories from the model
        batch = []
        running = 0

        active = [i for i in range(len(recv_sample))]

        while len(active) > 0:
            graph_batch = []
            graph_batch_arg = []
            graph_batch_proc = []
            rem_active_idx = []
            for i, idx in list(enumerate(active)):
                recv = recv_sample[idx].recv()
                data_type, data = pickle.loads(recv)

                if data_type == 0:
                    batch.append(data)
                    if len(batch) + len(active) - 1 < num_traj:
                        # We still need trajectories
                        recv = recv_sample[idx].recv()
                        data_type, data = pickle.loads(recv)
                    else:
                        # we need to remove proc from active
                        rem_active_idx.append(i)

                if data_type == 1:
                    graph_batch.append(data[0])
                    graph_batch_arg.append(data[1])
                    graph_batch_proc.append(idx)

            if len(graph_batch) > 0:
                gbatch_cnt = [len(x) for x in graph_batch]
                states = self.mdp.mols2batch([mol for mols in graph_batch for mol in mols])

                with torch.no_grad():
                    s_o, m_o = self.sampling_model(states)
                    s_o, m_o = s_o.cpu(), m_o.cpu()

                prev = 0
                stem_idx = states.__slices__["stems"]
                # TODO should implement batch sampling
                for iii, (cnt, proc, no_stop) in enumerate(zip(gbatch_cnt, graph_batch_proc, graph_batch_arg)):
                    # assume 1 cnt
                    send = s_o[stem_idx[prev]: stem_idx[prev+cnt]], m_o[prev: prev+cnt]

                    s_s_o, s_m_o = send
                    if no_stop:
                        s_m_o = s_m_o * 0 - 1000  # prevent assigning prob to stop when we can't stop

                    logits = torch.cat([s_m_o.reshape(-1), s_s_o.reshape(-1)])
                    if self.random_action_temp != 1:
                        logits = logits / self.random_action_temp

                    cat = torch.distributions.Categorical(logits=logits)
                    action = cat.sample().item()
                    send = action

                    prev += cnt
                    send_sample[proc].send(send)

            for i_rm in rem_active_idx[::-1]:
                active.pop(i_rm)

        # -- Recalculate Reward
        # # TODO Adapt with memory & maybe improve
        batch = batch[:num_traj]
        final_ms = [x[1][1] for x in batch]
        res_scores, infos = self.proxy_reward(final_ms)
        res_scores = self.r2r(dockscore=np.array(res_scores, dtype=np.float64))

        transf_batch = []
        shuffle_zips = list(zip(batch, res_scores, infos))
        random.shuffle(shuffle_zips)  # Usually List will end with longer trajectories
        for traj, res_s, r_info in shuffle_zips:
            samples, [r, m, trajectory_stats, inflow, info] = traj
            info.update(r_info)
            samples[-1].r = res_s
            traj = samples, [res_s, m, trajectory_stats, inflow, info]
            transf_batch.append(traj)
            self._add_new_trajectory(traj)

        batch = transf_batch
        return batch

    def sample2batch(self, samples: List[Any]):
        """ Just parallelize with multiple procs """
        p, a, r, s, d = list(zip(*samples))

        mols = (p, s)
        # The batch index of each parent
        p_batch = torch.tensor(sum([[i]*len(p) for i,p in enumerate(p)], []),
                               device=self._device).long()

        # Convert all parents and states to repr. Note that this
        # concatenates all the parent lists, which is why we need
        # p_batch

        # # Send to multiprocs
        mol_p = sum(p, ())
        mol_s = s

        _send_batch = list(mol_p) + list(mol_s)
        for _id, _mol in enumerate(_send_batch):
            self._repr_send.put((_id, pickle.dumps(_mol, -1)))  # Pickle improves a lot here!

        recv_count = 0
        while recv_count < len(_send_batch):
            _id, _molr = self._repr_recv.get()
            _send_batch[_id] = pickle.loads(_molr)  # Pickle improves a lot here!
            recv_count += 1

        mol_p = _send_batch[:len(mol_p)]
        mol_s = _send_batch[len(mol_p):]

        p = self.mdp.mols2batch(mol_p)
        s = self.mdp.mols2batch(mol_s)

        # Concatenate all the actions (one per parent per sample)
        a = torch.tensor(sum(a, ()), device=self._device).long()
        # rewards and dones
        r = torch.tensor(r, device=self._device).to(self.floatX)
        d = torch.tensor(d, device=self._device).to(self.floatX)

        return TrainBatch(p, p_batch, a, r, s, d, mols)

    def stop_samplers_and_join(self):
        for p in self._sample_processes:
            p.terminate()
            p.join()
        for p in self._repr_procs:
            p.terminate()
            p.join()
