import argparse
from copy import copy, deepcopy
from collections import defaultdict
import gc
import gzip
import os
import os.path as osp
import pickle
import psutil
import pdb
import subprocess
import sys
import threading
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
if __name__ == "__main__":
  mp.set_start_method('spawn')
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

import LambdaZero.models
from LambdaZero import chem
from LambdaZero.chem import chem_op
from LambdaZero.environments.molMDP import BlockMoleculeData, MolMDP
#from LambdaZero.environments.reward import PredDockReward
#from LambdaZero.environments.reward import PredDockReward_v3 as PredReward
from LambdaZero.environments.block_mol_v3 import DEFAULT_CONFIG as env_v3_cfg, BlockMolEnv_v3
from LambdaZero.examples.synthesizability.vanilla_chemprop import synth_config, binding_config
from LambdaZero.utils import get_external_dirs


from LambdaZero.examples.goal_based_imitation import model_atom, model_block, model_fingerprint

import importlib
importlib.reload(model_block)
importlib.reload(chem_op)

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
tmp_dir = os.environ['SLURM_TMPDIR'] + '/lztmp/'
if 0:
    print("Syncing locally")

    os.makedirs(tmp_dir, exist_ok=True)
    os.system(f"rsync -az {programs_dir} {tmp_dir}")
    os.system(f"rsync -az {datasets_dir} {tmp_dir}")
programs_dir = f"{tmp_dir}/Programs"
datasets_dir = f"{tmp_dir}/Datasets"
print("Done syncing")

parser = argparse.ArgumentParser()

parser.add_argument("--learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=32, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--nemb", default=32, help="#hidden", type=int)
parser.add_argument("--num_iterations", default=100000, type=int)
parser.add_argument("--num_conv_steps", default=6, type=int)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v1')
parser.add_argument("--run", default=0, help="run", type=int)
parser.add_argument("--save_path", default='/miniscratch/bengioe/LambdaZero/imitation_learning/')
parser.add_argument("--print_array_length", default=False, action='store_true')
parser.add_argument("--dump_episodes", default='')
parser.add_argument("--gen_rand", default='')



class BlockMoleculeDataNoCache(BlockMoleculeData):

    def restore_blocks(self, bdata):
        self.blocks = [bdata.block_mols[i] for i in self.blockidxs]

    @property
    def mol(self):
        return chem.mol_from_frag(jun_bonds=self.jbonds, frags=self.blocks)[0]

    def copy(self, no_blocks=False): # shallow copy
        o = BlockMoleculeDataNoCache()
        o.blockidxs = list(self.blockidxs)
        if no_blocks:
            o.blocks = None
        else:
            o.blocks = list(self.blocks)
        o.slices = list(self.slices)
        o.numblocks = self.numblocks
        o.jbonds = list(self.jbonds)
        o.stems = list(self.stems)
        return o

class BlocksData:
    def __init__(self, config):
        blocks = pd.read_json(config['blocks_file'])
        self.block_smi = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in blocks["block_smi"]]
        self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
        self.num_blocks = len(self.block_smi)



class SimDockLet:
    def __init__(self, tmp_dir):
        self.dock = chem.Dock_smi(tmp_dir,
                                  osp.join(programs_dir, 'chimera'),
                                  osp.join(programs_dir, 'dock6'),
                                  osp.join(datasets_dir, 'brutal_dock/mpro_6lze/docksetup'),
                                  gas_charge=True)
        self.target_norm = binding_config['dockscore_std'] #[-26.3, 12.3]

    def eval(self, mol):
        #print("starting", s)
        s = "None"
        try:
            s = Chem.MolToSmiles(mol.mol)
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        #print("done", s, r)
        return reward


class MolMDPNC(MolMDP):

    def add_block(self, block_idx, stem_idx):
        if len(self.molecule.stems) == 0 and False: # Why was this here??
            if len(self.molecule.blocks) != 1:
                print("Molecule with no stems is being added to?")
                print(self.molecule.blockidxs, block_idx, stem_idx)
                return
            stem_idx = None
        try:
            super().add_block(block_idx, stem_idx)
        except:
            print("Failed to add block", block_idx, stem_idx)
            print(self.molecule.blockidxs, self.molecule.jbonds, self.molecule.stems)

    def add_block_to(self, mol, block_idx, stem_idx=None, atmidx=None):
        assert (block_idx >= 0) and (block_idx <= len(self.block_mols)), "unknown block"
        if mol.numblocks == 0:
            stem_idx = None
        new_mol = mol.copy()
        new_mol.add_block(block_idx,
                          block=self.block_mols[block_idx],
                          block_r=self.block_rs[block_idx],
                          stem_idx=stem_idx, atmidx=atmidx)
        return new_mol

    def reset(self):
        self.molecule = BlockMoleculeDataNoCache()
        return None

    def post_init(self, device, repr_type):
        self.device = device
        self.repr_type = repr_type
        #self.max_bond_atmidx = max([max(i) for i in self.block_rs])
        self.max_num_atm = max(self.block_natm)
        self.stem_type_offset = np.int32([0] + list(np.cumsum([max(i)+1 for i in self.block_rs])))
        self.num_stem_types = self.stem_type_offset[-1]
        #print(self.max_num_atm, self.num_stem_types)
        self.molcache = {}

    def mols2batch(self, mols):
        if self.repr_type == 'block_graph':
            return model_block.mols2batch(mols, self)
        elif self.repr_type == 'atom_graph':
            return model_atom.mols2batch(mols, self)
        elif self.repr_type == 'morgan_fingerprint':
            return model_fingerprint.mols2batch(mols, self)

    def mol2repr(self, mol=None):
        if mol is None:
            mol = self.molecule
        molhash = str(mol.blockidxs)+':'+str(mol.stems)+':'+str(mol.jbonds)
        if molhash in self.molcache:
            return self.molcache[molhash]
        if self.repr_type == 'block_graph':
            r = model_block.mol2graph(mol, self)
        elif self.repr_type == 'atom_graph':
            r = model_atom.mol2graph(mol, self)
        elif self.repr_type == 'morgan_fingerprint':
            r = model_fingerprint.mol2fp(mol, self)
        self.molcache[molhash] = r
        return r



class MolEpisode:
    def __init__(self, actions, rewards, goal, reached, time=0, gtype=None):
        self.actions = actions
        self.rewards = np.float32(rewards) # the state that was reached
        self.goal = goal # the goal given to the agent
        self.reached = reached # 1 if goal ~ rewards
        self.gdist = ((np.float32(goal)-self.rewards)**2).mean()
        self.time = time # how many training steps/training time has
                         # the agent generating this mol had
        self.gtype = gtype # The type of generating episode


    def sample(self, mdp, t=None):
        if t is None:
            t = np.random.randint(0, len(self.actions))
        mdp.reset()
        for i in self.actions[:t]:
            if None in i:
                print('wtf', self.actions)
                return self.sample(mdp)
            if i[0] >= 0:
                mdp.add_block(*i)
            else:
                break
        s = mdp.molecule.copy()
        # 3.356 is the median (normalized) energy of the raw_dump_2021_01_07 dataset
        # so this splits the dataset in half
        two_bin_goal = np.float32([1,0,0]) if self.rewards[0] < 3.356 else np.float32([0,1,0])
        return s, self.actions[t], two_bin_goal#self.rewards

    def iterate(self, mdp):
        for i in range(len(self.actions)):
            yield self.sample(mdp, t=i)



def mp_sampler_fn(pid, mol_queue, graph_queue, mdp, stop_event):
    #print("sampler started", pid)
    #print(">>", mol_queue.qsize())
    #print(">>")
    while not stop_event.is_set():
        #t0 = time.time()
        s, a, g = mol_queue.get()
        #t1 = time.time()
        #print(f'waited mol queue {t1-t0:.4f}',mol_queue.qsize())
        s = mdp.mols2batch([mdp.mol2repr(i) for i in s])
        a = torch.tensor(a, device=mdp.device).long()
        g = torch.tensor(g, device=mdp.device).float()
        graph_queue.put((s, a, g))
    #print("sampler done")


class ReplayBuffer:

    def __init__(self, bpath, device, repr_type):
        self.test_split_rng = np.random.RandomState(142857)
        self.episodes = []
        self.test_episodes = []
        self.mdp = MolMDPNC(bpath)
        self.mdp.post_init(device, repr_type)
        self._device = device
        self.epsilon = 0.05
        self.seen_molecules = set()
        self._mahash = lambda actions: '.'.join(','.join(map(str,i)) for i in actions)

    def add_episode(self, actions, rewards, goal, reached, time=0, gtype=None):
        self.seen_molecules.add(self._mahash(actions))
        self.episodes.append(MolEpisode(actions, rewards, goal, reached, time, gtype))

    def contains(self, actions):
        return self._mahash(actions) in self.seen_molecules

    def _iterate_test_set(self):
        for ep in self.test_episodes:
            for i in ep.iterate(self.mdp):
                yield i

    def iterate_test(self, n):
        ts = self._iterate_test_set()
        while True:
            s = [i for _, i in zip(range(n), ts)]
            if len(s) == 0:
                break
            yield self.sample2batch(zip(*s))
            if len(s) < n:
                break

    def sample(self, n):
        eidx = np.random.randint(0, len(self.episodes), n)
        try:
            samples = [self.episodes[i].sample(self.mdp) for i in eidx]
        except:
            return self.sample(n) # idk... weird shit happens randomly, I don't want to deal with it
        return zip(*samples)

    def sample2batch(self, mb):
        s, a, g = mb
        s = self.mdp.mols2batch([self.mdp.mol2repr(i) for i in s])
        a = torch.tensor(a, device=self._device).long()
        g = torch.tensor(g, device=self._device).float()
        return s, a, g

    def load_raw_episodes(self, episodes, test_ratio=0.02):
        for acts, rewards in episodes:
            if self.contains(acts):
                continue
            self.seen_molecules.add(self._mahash(acts))
            ep = MolEpisode(acts, rewards, rewards, True)
            if self.test_split_rng.uniform() < test_ratio:
                self.test_episodes.append(ep)
            else:
                self.episodes.append(ep)
        self.num_loaded_episodes = len(self.episodes)

    def load_initial_episodes(self, episodes, test_ratio=0.05):
        for i in episodes:
            if self.contains(i.actions):
                continue
            self.seen_molecules.add(self._mahash(i.actions))
            if self.test_split_rng.uniform() < test_ratio:
                self.test_episodes.append(i)
            else:
                self.episodes.append(i)
        self.num_loaded_episodes = len(self.episodes)

    def start_samplers_mp(self, n, mbsize):
        self.mp_mol_queue = mp.Queue(32)
        self.mp_graph_queue = mp.Queue(32)
        self.mp_stop_event = mp.Event()
        self.procs = [
            mp.spawn(mp_sampler_fn, (self.mp_mol_queue, self.mp_graph_queue,
                                     self.mdp, self.mp_stop_event),
                     join=False)
            for i in range(n)]
        def mol_queue_filler_thread():
            while not self.mp_stop_event.is_set():
                mol_batch = self.sample(mbsize)
                self.mp_mol_queue.put(mol_batch)
                #print("put mol_batch")

        self.mp_filler_thread = threading.Thread(target=mol_queue_filler_thread, args=())
        self.mp_filler_thread.start()
        time.sleep(0.5)
        def _gen():
            while not self.mp_stop_event.is_set():
                #print('main qsize', self.mp_mol_queue.qsize())
                #t0 = time.time()
                x = self.mp_graph_queue.get()
                #t1 = time.time()
                #print(f'waited graph queue {t1-t0:.4f}', self.mp_graph_queue.qsize())
                yield x
                del x
        gen = _gen()

        def get():
            return next(gen)
        return get



    def start_samplers(self, n, mbsize):
        self.ready_events = [threading.Event() for i in range(n)]
        self.resume_events = [threading.Event() for i in range(n)]
        self.results = [None] * n
        self.stop_event = threading.Event()
        def f(idx):
            while not self.stop_event.is_set():
                try:
                    self.results[idx] = self.sample2batch(self.sample(mbsize))
                except Exception as e:
                    print("Exception while sampling:")
                    print(e)
                    continue
                self.ready_events[idx].set()
                self.resume_events[idx].clear()
                self.resume_events[idx].wait()
        self.sampler_threads = [threading.Thread(target=f, args=(i,)) for i in range(n)]
        [i.start() for i in self.sampler_threads]
        round_robin_idx = [0]
        def get():
            while True:
                idx = round_robin_idx[0]
                round_robin_idx[0] = (round_robin_idx[0] + 1) % n
                if self.ready_events[idx].is_set():
                    r = self.results[idx]
                    self.ready_events[idx].clear()
                    self.resume_events[idx].set()
                    return r
                elif round_robin_idx[0] == 0:
                    time.sleep(0.001)
        return get

    def stop_samplers_and_join(self):
        self.stop_event.set()
        while any([i.is_alive() for i in self.sampler_threads]):
            [i.set() for i in self.resume_events]
            [i.join(0.05) for i in self.sampler_threads]

    def dump_to_raw(self):
        raw = []
        for e in self.episodes:
            raw.append((e.actions, [float(i) for i in e.rewards]))
        return raw

class RolloutActor(threading.Thread):
    def __init__(self, bpath, device, repr_type, model, stop_event, replay, synth_net, sample_type,
                 stop_after=0, greedy=False):
        super().__init__()
        self.device = torch.device(device)
        self.mdp = MolMDPNC(bpath)
        self.mdp.post_init(self.device, repr_type)
        self.max_blocks = 5, 13
        # energy, synth, qed
        self.goal_bounds = np.float32([[0, 8], [0.1, 1], [0.1, 1]]).T
        self.greedy_goal_bounds = np.float32([[6, 8], [0.5, 1], [0.7, 1]]).T
        if greedy:
            self.goal_bounds = self.greedy_goal_bounds
        self.model = model
        self.stop_event = stop_event
        self.docker = SimDockLet(tmp_dir)
        self.synth_net = synth_net
        self.replay = replay
        self.failed = False
        self.sample_type = sample_type
        self.beam_max = 6
        self.stop_after = stop_after
        self.episodes_done = 0


    def run(self):
        while not self.stop_event.is_set():
            if self.sample_type == 'categorical':
                self._sample_episode()
            elif self.sample_type == 'beam':
                self._beam_episode()
            elif self.sample_type == 'random':
                self._random_episode()
            self.episodes_done += 1
            if self.stop_after > 0 and self.episodes_done >= self.stop_after:
                break


    def _random_episode(self):
        self.mdp.reset()
        actions = []
        done = False
        goal = (torch.tensor(np.random.uniform(*self.goal_bounds))
                .reshape((1,3)).float().to(self.device))
        max_len = np.random.randint(*self.max_blocks)
        while not done:
            if (len(self.mdp.molecule.blocks) >= max_len or
                len(self.mdp.molecule.blocks) > 0 and len(self.mdp.molecule.stems) == 0):
                actions.append((-1, 0))
                break
            action = np.random.randint(
                0, 1 + self.mdp.num_blocks * max(len(self.mdp.molecule.stems), 1))
            if action == 0:
                done = True
                action = (-1, 0)
            else:
                action -= 1
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                self.mdp.add_block(*action)
            actions.append(action)
        rdmol = self.mdp.molecule.mol
        if self.replay.contains(actions): return
        if rdmol is None: return
        energy = self.docker.eval(self.mdp.molecule)
        synth = self.synth_net(mol=rdmol) / 10
        qed = QED.qed(rdmol)
        goal = goal.cpu().numpy()[0]
        reached_state = np.float32((energy, synth, qed))
        #print(reached_state, goal)
        goal_reached = float(((reached_state - goal)**2).sum() < 0.05)
        self.replay.add_episode(actions, reached_state, goal, goal_reached,
                                self.model.training_steps if self.model is not None else 0,
                                'random')

    def _sample_episode(self):
        self.mdp.reset()
        actions = []
        done = False
        goal = (torch.tensor(
            np.float32([1,0,0]) if np.random.uniform() < 0.5 else np.float32([0,1,0]))
        #goal = (torch.tensor(np.random.uniform(*self.goal_bounds))
                .reshape((1,3)).float().to(self.device))
        max_len = np.random.randint(*self.max_blocks)
        while not done:
            if (len(self.mdp.molecule.blocks) >= max_len or
                len(self.mdp.molecule.blocks) > 0 and len(self.mdp.molecule.stems) == 0):
                actions.append((-1, 0))
                break
            batch = self.mdp.mols2batch([self.mdp.mol2repr()])
            stem_o, stop_o = self.model(batch, goal)
            policy = torch.cat([i.reshape(-1)
                                for i in self.model.out_to_policy(batch, stem_o, stop_o)])
            #logits = torch.cat([stop_o.reshape(-1), stem_o.reshape(-1)])
            if torch.isnan(policy).any():
                self.failed = True
                self.tb = traceback.extract_stack()
                raise ValueError("nan found")
            #action = F.softmax(logits).multinomial(1).item()
            action = policy.multinomial(1).item()
            if action == 0:
                done = True
                action = (-1, 0)
            else:
                action -= 1
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                self.mdp.add_block(*action)
            actions.append(action)
        rdmol = self.mdp.molecule.mol
        if self.replay.contains(actions): return
        if rdmol is None: return
        energy = self.docker.eval(self.mdp.molecule)
        synth = self.synth_net(mol=rdmol) / 10
        qed = QED.qed(rdmol)
        goal = goal.cpu().numpy()[0]
        reached_state = np.float32((energy, synth, qed))
        print(reached_state, goal)
        goal_reached = float(((reached_state - goal)**2).sum() < 0.05)
        self.replay.add_episode(actions, reached_state, goal, goal_reached,
                                self.model.training_steps, 'sample')


    def _beam_episode(self):
        self.mdp.reset()
        done = []
        goal = (torch.tensor(np.random.uniform(*self.greedy_goal_bounds))
                .reshape((1,3)).float().to(self.device))
        beam = [(self.mdp.molecule.copy(), 0, [])]
        actions = []
        while len(done) < self.beam_max:
            if not len(beam): # Unlikely ?
                break
            mol, logprob, actions = beam.pop(0)
            if (len(mol.blocks) >= self.max_blocks[1] - 1 or
                len(mol.blocks) > 0 and len(mol.stems) == 0 or
                (len(actions) and actions[-1][0] == -1)):
                if mol.mol is None: # Ignore invalid molecules
                    continue
                if self.replay.contains(actions): # Ignore previously seen molecules
                    continue
                if len(actions) and actions[-1][0] != -1:
                    actions += [(-1, 0)]
                done.append((mol, logprob, actions))
                continue

            batch = self.mdp.mols2batch([self.mdp.mol2repr()])
            stem_o, stop_o = self.model(batch, goal)
            logits = F.log_softmax(torch.cat([stop_o.reshape(-1), stem_o.reshape(-1)]), 0)
            if torch.isnan(logits).any():
                self.failed = True
                self.tb = traceback.extract_stack()
                raise ValueError("nan found")
            for i in range(logits.shape[0]):
                if i == 0:
                    action = ((-1, 0))
                    new_mol = mol
                else:
                    action = ((i - 1) % self.mdp.num_blocks, (i - 1) // self.mdp.num_blocks)
                    new_mol = self.mdp.add_block_to(mol, *action)
                new_logprob = logprob + logits[i].item()
                new_actions = actions + [action]
                beam.append((new_mol, new_logprob, new_actions))
            beam = sorted(beam, key=lambda x: -x[1])
        print(' '.join(f'{i[1]:.2f}' for i in done))
        goal = goal.cpu().numpy()[0]
        for mol, lp, actions in done:
            rdmol = mol.mol
            if rdmol is None: continue
            qed = QED.qed(rdmol)
            synth = self.synth_net(mol=rdmol) / 10
            if qed < 0.2 or synth < 0.2: # Avoid docking ridiculous molecules
                continue # I'm not sure I should be doing this, but
                         # sometimes docking takes forever, because the
                         # molecules seem absurd.
            energy = self.docker.eval(mol)
            reached_state = np.float32((energy, synth, qed))
            print('beam', reached_state, goal, lp)
            goal_reached = float(((reached_state - goal)**2).sum() < 0.05)
            self.replay.add_episode(actions, reached_state, goal, goal_reached,
                                    self.model.training_steps, 'beam')




def dump_episodes(args):
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")

    device = torch.device('cpu')
    mdp = MolMDPNC(bpath)
    mdp.post_init(device, None)
    replay = ReplayBuffer(bpath, device, None)
    past_ep = pickle.load(gzip.open('replays/random_15000.pkl.gz', 'rb'))
    replay.load_initial_episodes(past_ep, test_ratio=0)
    print(replay.num_loaded_episodes)
    past_ep = pickle.load(gzip.open('raw_dump.pkl.gz', 'rb'))
    replay.load_raw_episodes(past_ep, test_ratio=0)
    print(replay.num_loaded_episodes)
    raw = replay.dump_to_raw()
    with gzip.open('raw_dump_2021_01_07.pkl.gz','wb') as f:
        pickle.dump(raw, f)


def generate_random_episodes(args):
    device = torch.device('cuda')
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

    replay = ReplayBuffer(bpath, device, args.repr_type)
    stop_event = threading.Event()
    rollout_threads = (
        [RolloutActor(bpath, device, None, None,
                      stop_event, replay, synth_net, 'random')
         for i in range(8)])
    [i.start() for i in rollout_threads]

    while len(replay.episodes) < 50000:
        time.sleep(30)
        print('>>', len(replay.episodes))
        pickle.dump(replay.episodes, gzip.open('replays/random_50000.pkl.gz', 'wb'))
    stop_event.set()
    [i.join() for i in rollout_threads]
    pickle.dump(replay.episodes, gzip.open('replays/random_50000.pkl.gz', 'wb'))
    return

def main(args):
    debug_no_threads = False

    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")

    device = torch.device('cuda')
    mdp = MolMDPNC(bpath)
    mdp.post_init(device, args.repr_type)

    synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

    exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
    os.makedirs(exp_dir, exist_ok=True)


    replay = ReplayBuffer(bpath, device, args.repr_type)
    #replay.epsilon = 0.95

    #past_ep = pickle.load(gzip.open('replays/random_1000.pkl.gz', 'rb'))
    # This includes the tree search mols, to be seen...
    past_ep = pickle.load(gzip.open('raw_dump_2021_01_07.pkl.gz', 'rb'))
    replay.load_raw_episodes(past_ep)
    #past_ep = pickle.load(gzip.open('replays/random_15000.pkl.gz', 'rb'))
    #replay.load_initial_episodes(past_ep, test_ratio=0.05)
    #past_ep = pickle.load(gzip.open('replays/latest_oct_22_5.pkl.gz', 'rb'))
    #replay.load_initial_episodes(past_ep)
    print(len(replay.episodes), 'episodes')


    stop_event = threading.Event()

    if args.repr_type == 'block_graph':
        model = model_block.GraphAgent(args.nemb, 3, mdp.num_blocks, 1, args.num_conv_steps, mdp,
                                       args.model_version)
        model.to(device)
    elif args.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(args.nemb, 3, mdp.num_blocks, 1)
        model.to(device)
    elif args.repr_type == 'morgan_fingerprint':
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)
        model.to(device)

    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-4,
                           betas=(args.opt_beta, 0.999))

    rollout_threads = (
        [RolloutActor(bpath, device, args.repr_type, model,
                      stop_event, replay, synth_net, 'categorical')
         for i in range(0)] +
        [RolloutActor(bpath, device, args.repr_type, model,
                      stop_event, replay, synth_net, 'beam')
         for i in range(0)])
    if not debug_no_threads:
        [i.start() for i in rollout_threads]

    tf = lambda x: torch.tensor(x, device=device).float()
    tint = lambda x: torch.tensor(x, device=device).long()
    mbsize = args.mbsize
    ar = torch.arange(mbsize)
    if not debug_no_threads:
        sampler = replay.start_samplers(4, mbsize)
    gamma = 0.99
    last_losses = []

    def stop_everything():
        stop_event.set()
        print('joining')
        [i.join() for i in rollout_threads]
        replay.stop_samplers_and_join()

    def save_stuff():
        pickle.dump(replay.episodes[replay.num_loaded_episodes:],
                    gzip.open(f'{exp_dir}/replay.pkl.gz', 'wb'))
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

    train_losses = []
    test_losses = []
    time_start = time.time()
    time_last_check = time.time()

    for i in range(args.num_iterations+1):
        if not debug_no_threads:
            s, a, g = sampler()
        else:
            s, a, g = replay.sample2batch(replay.sample(mbsize))
        stem_o, mol_o = model(s, g)
        negloglike = model.action_negloglikelihood(s, a, g, stem_o, mol_o).mean()
        logit_reg = (mol_o.pow(2).sum() + stem_o.pow(2).sum()) / (np.prod(mol_o.shape) + np.prod(stem_o.shape))

        #(negloglike + logit_reg * 1e-1).backward()
        negloglike.backward()
        last_losses.append(negloglike.item())
        train_losses.append(negloglike.item())
        opt.step()
        opt.zero_grad()
        model.training_steps = i + 1
        #if any([torch.isnan(i).any() for i in model.parameters()]):
        if torch.isnan(negloglike):
            stop_event.set()
            stop_everything()
            raise ValueError()
            #pdb.set_trace()
        for thread in rollout_threads:
            if thread.failed:
                stop_event.set()
                pdb.set_trace()

        if not i % 250:
            #recent_best = np.argmax([i.rewards[0] * max(0,i.rewards[1]) * i.rewards[2] for i in replay.episodes[-100:]])
            #recent_best_m = np.max([i.rewards[0] * max(0,i.rewards[1]) * i.rewards[2] for i in replay.episodes[-100:]])
            last_losses = np.mean(last_losses)
            print(i, last_losses)#, np.mean([i.gdist for i in replay.episodes[-100:]]),
            #recent_best_m, replay.episodes[-100+recent_best].rewards)
            #print(logit_reg)
            #print(stem_o.mean(), stem_o.max(), stem_o.min())
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

            if 0:
                qed_cutoff = [0.2, 0.7]
                synth_cutoff = [0, 0.4]
                all_rs = []
                for ep in replay.episodes[replay.num_loaded_episodes:]:
                    nrg, synth, qed = ep.rewards
                    qed_discount = (qed - qed_cutoff[0]) / (qed_cutoff[1] - qed_cutoff[0])
                    qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1
                    synth_discount = (synth - synth_cutoff[0]) / (synth_cutoff[1] - synth_cutoff[0])
                    synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1
                    all_rs.append(nrg * synth_discount * qed_discount)
                print(np.mean(all_rs), sorted(all_rs)[-10:])

            if i % 5000:
                continue

            t0 = time.time()
            total_test_loss = 0
            total_test_n = 0
            for s, a, g in replay.iterate_test(mbsize):
                with torch.no_grad():
                    stem_o, mol_o = model(s, g)
                total_test_loss += model.action_negloglikelihood(s, a, g, stem_o, mol_o).sum().item()
                total_test_n += g.shape[0]
            test_nll = total_test_loss / total_test_n
            print('test NLL:', test_nll)
            print('test took:', time.time() - t0)
            test_losses.append(test_nll)
            save_stuff()


    stop_everything()


    print("Running final search")
    stop_event = threading.Event()
    rollout_threads = (
        [RolloutActor(bpath, device, args.repr_type, model,
                      stop_event, replay, synth_net, 'categorical',
                      greedy=True,
                      stop_after=5)
         for i in range(8)]
        +
        [RolloutActor(bpath, device, args.repr_type, model,
                      stop_event, replay, synth_net, 'categorical',
                      greedy=False,
                      stop_after=5)
         for i in range(8)])
    if not debug_no_threads:
        [i.start() for i in rollout_threads]
        [i.join() for i in rollout_threads]
    save_stuff()
    print('Done.')


def array_nov_3(args):
  all_hps = ([
    {'mbsize': 32,
     'learning_rate': lr,
     'num_iterations': 200_000,
     'nemb': nemb,
     }
    for lr in [1e-4, 1e-3, 1e-5]
    for nemb in [16, 32, 64]
  ])
  return all_hps

def array_nov_25(args):
  all_hps = ([
    {'mbsize': 32,
     'learning_rate': lr,
     'num_iterations': 200_000,
     'nemb': nemb,
     'repr_type': repr_type,
     }
    for lr in [1e-4, 1e-3]
    for nemb in [16, 32, 64]
    for repr_type in ['block_graph', 'atom_graph']
  ])
  return all_hps

def array_dec_02(args):
  all_hps = ([
    {'mbsize': 32,
     'learning_rate': lr,
     'num_iterations': 200_000,
     'nemb': nemb,
     'repr_type': repr_type,
     }
    for lr in [1e-4, 1e-3]
    for nemb in [16, 32, 64]
    for repr_type in ['morgan_fingerprint']
  ])
  return all_hps

def array_dec_08(args):
  all_hps = ([
    {'mbsize': 32,
     'learning_rate': lr,
     'num_iterations': 200_000,
     'nemb': nemb,
     'repr_type': repr_type,
     'model_version': 'v2',
     }
    for lr in [1e-4, 1e-3]
    for nemb in [16, 32, 64]
    for repr_type in ['block_graph']
  ])
  return all_hps

def array_jan_07(args):
  all_hps = ([
    {'mbsize': 512,
     'learning_rate': lr,
     'num_iterations': 300_000,
     'nemb': nemb,
     'repr_type': repr_type,
     'model_version': 'v2',
     }
    for lr in [1e-3]
    for nemb in [128, 256]
    for repr_type in ['block_graph', 'morgan_fingerprint']
  ])
  all_hps += [
      {**all_hps[0], 'nemb': 256, 'repr_type': 'morgan_fingerprint', 'learning_rate': 1e-4},
      {**all_hps[0], 'nemb': 128, 'repr_type': 'morgan_fingerprint', 'learning_rate': 1e-4},
      {**all_hps[0], 'nemb': 128, 'repr_type': 'morgan_fingerprint', 'learning_rate': 1e-4}, ## 15k r
      {**all_hps[0], 'nemb': 128, 'repr_type': 'morgan_fingerprint', 'learning_rate': 1e-4, # 7
       'num_iterations': 200000},
      {**all_hps[0], 'learning_rate': 5e-4,
       'num_iterations': 200000},
      {**all_hps[0], 'learning_rate': 5e-4, 'num_conv_steps': 12, 'nemb': 64,
       'num_iterations': 200000},
      {**all_hps[0], 'learning_rate': 1e-3, 'num_conv_steps': 12, 'nemb': 64,
       'num_iterations': 200000},
      {**all_hps[0], 'learning_rate': 1e-3, 'num_conv_steps': 12, 'nemb': 64, # 11
       'num_iterations': 200000}, # escort p 6
      {**all_hps[0], 'learning_rate': 1e-3, 'num_conv_steps': 3, 'nemb': 64,
       'num_iterations': 200000, 'model_version': 'v3'}, # escort p 8, v3
      {**all_hps[0], 'learning_rate': 1e-3, 'num_conv_steps': 3, 'nemb': 64,
       'num_iterations': 200000, 'model_version': 'v3'}, # escort p 8, v3
      {**all_hps[0], 'learning_rate': 2e-3, 'num_conv_steps': 12, 'nemb': 64,
       'opt_beta': 0.95,
       'num_iterations': 500000}, # 14, escort p 6
      {**all_hps[0], 'learning_rate': 1e-3, 'num_conv_steps': 20, 'nemb': 64,
       'num_iterations': 500000}, # escort p 6
      {**all_hps[0], 'learning_rate': 1e-4, 'num_conv_steps': 20, 'nemb': 64,
       'num_iterations': 500000}, # escort p 6
      {**all_hps[0], 'learning_rate': 2e-4, 'num_conv_steps': 20, 'nemb': 64,
       'num_iterations': 500000}, # escort p 6
      {**all_hps[0], 'learning_rate': 5e-4, 'num_conv_steps': 20, 'nemb': 64,
       'num_iterations': 500000}, # escort p 6
  ]
  return all_hps

if __name__ == '__main__':
  args = parser.parse_args()
  if args.array:
    all_hps = eval(args.array)(args)

    if args.print_array_length:
      print(len(all_hps))
      #print(','.join(str(i) for i, h in enumerate(all_hps) if h['opt'] == 'msgd_corr'))
      #print(' '.join(f'run_{i}' for i, h in enumerate(all_hps) if h['opt'] == 'msgd_corr'))
    else:
      hps = all_hps[args.run]
      for k,v in hps.items():
        setattr(args, k, v)
      main(args)
  elif args.dump_episodes:
      dump_episodes(args)
  elif args.gen_rand:
      generate_random_episodes(args)
  else:
    main(args)
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu102.html
pip install torch-geometric
