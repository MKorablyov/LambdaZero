from copy import copy, deepcopy
from collections import defaultdict
import gc
import gzip
import os
import os.path as osp
import pickle
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import ray
from rdkit import Chem
from rdkit.Chem import QED
import torch
from torch_geometric.data import Data, Batch


import warnings
warnings.filterwarnings('ignore')

from LambdaZero.environments.persistent_search.fast_sumtree import SumTree
import LambdaZero.models
from LambdaZero import chem
from LambdaZero.environments.molMDP import BlockMoleculeData
from LambdaZero.environments.reward import PredDockReward
from LambdaZero.utils import Complete, get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

import psutil



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


class Node:
    def __init__(self, id, mol, n_legal_acts, parent):
        self.id = id
        self.mol = mol
        self.value = 0
        self.fast_reward = -0.5
        self.qed = None
        self.synth = None
        self.pred_dock_reward = None
        self.sim_dock_reward = None
        self.children = {}
        self.pruned_children = {}
        self.parent = parent
        self.n_legal_acts = n_legal_acts
        self.max_descendant_r = -0.5
        self.total_return = 0
        self.total_leaves = 1

    @property
    def montecarlo_return(self):
        return self.total_return / self.total_leaves

    def discounted_reward(self, reward):
        qed = self.qed[1] if self.qed is not None else 1
        synth = self.synth[1] if self.synth is not None else 1
        return min(reward, reward * qed * synth)

    @property
    def reward(self):
        if self.sim_dock_reward is not None:
            return self.discounted_reward(self.sim_dock_reward)
        if self.pred_dock_reward is not None:
            r = self.discounted_reward(self.pred_dock_reward)
            rp = max(self.fast_reward, r) # optimistic
            return rp if rp > 0 else r
        return self.fast_reward


def find_kth(l, k):
    pivot = l[k]
    lt = [i for i in l if i < pivot]
    gt = [i for i in l if i > pivot]
    pt = [i for i in l if i == pivot]
    if k < len(lt):
        return find_kth(lt, k)
    elif k < len(lt) + len(pt):
        return pt[0]
    else:
        return find_kth(gt, k - len(pt) - len(lt))




class BlocksData:
    def __init__(self, config):
        blocks = pd.read_json(config['blocks_file'])
        self.block_smi = blocks["block_smi"].to_list()
        self.block_rs = blocks["block_r"].to_list()
        self.block_nrs = np.asarray([len(r) for r in self.block_rs])
        self.block_mols = [Chem.MolFromSmiles(smi) for smi in blocks["block_smi"]]
        self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
        self.num_blocks = len(self.block_smi)

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


class PersistentSearchTree:

    def __init__(self, config):
        self.bdata = BlocksData(config)
        self.num_blocks = len(self.bdata.block_smi)
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.num_actions = self.max_branches * self.num_blocks
        self.exp_path = config["exp_path"]
        self.return_type = config.get('return_type', 'max_desc_r')
        assert self.return_type in ['max_desc_r', 'montecarlo']
        self.update_prio_on_refresh = config.get('update_prio_on_refresh', False)

        self.root = Node(0, BlockMoleculeDataNoCache(), [], None)
        self.root.n_legal_acts = len(self.get_legal_actions(self.root.mol))
        self.score_fn = config['score_fn']
        self.max_size = config['num_molecules']

        self.nodes = {}
        self.nodes[0] = self.root
        self.nodes_lock = threading.Lock()
        self.sumtree = SumTree(self.max_size)
        self.num_explored_nodes = 1
        self.prune_factor = 0.25
        self.dummy_mol = self.bdata.add_block_to(self.bdata.add_block_to(self.root.mol, 0, 0), 0, 0)
        self.new_node_queue = []


        if 'seeding_nodes' in config:
            print("Loading seed file...")
            nodes = pickle.load(gzip.open(config['seeding_nodes'], 'rb'))
            print("Seeding tree", len(nodes))
            for i in nodes:
                n = self.root
                for a, r in zip(i[2], i[3]):
                    if a not in n.children:
                        self.take_action(n.id, a, precomputed=r)
                    n = self.nodes[n.children[a]]

        if config.get('populate_root', True):
            for action in self.get_legal_actions(self.root.mol):
                self.take_action(0, action)

    def get_mols(self, idxs, return_la_mask=False):
        mols = [self.nodes[i].mol if i in self.nodes else self.dummy_mol for i in idxs]
        if return_la_mask:
            mask = torch.tensor([self.get_legal_actions(i, return_mask=True) for i in mols])
            return mols, mask
        return mols

    def pop_from_new_queue(self):
        self.nodes_lock.acquire()
        r = self.new_node_queue
        self.new_node_queue = []
        self.nodes_lock.release()
        return r


    def sample_many(self, n, just_idxs=False, idxs_and_aa=False):
        q = np.random.uniform(0, 1, n)
        data = []
        while len(data) < n:
            idx = self.sumtree.sample(q[len(data)])
            if idx not in self.nodes:
                print('Not sampling empty node',idx)
                q[len(data)] = np.random.uniform(0, 1)
                self.sumtree.set(idx, 0)
                continue
            node = self.nodes[idx]
            la = self.get_legal_actions(node.mol)
            aa = np.int32([i for i in la if i not in node.children.keys()]) # available actions
            if not len(aa):
                q[len(data)] = np.random.uniform(0, 1)
                self.sumtree.set(idx, 0)
                continue

            if just_idxs:
                data.append(idx)
                continue
            if idxs_and_aa:
                data.append((idx, aa))
                continue
            if self.return_type == 'max_desc_r':
                target = {a: self.nodes[j].max_descendant_r
                          if a not in node.pruned_children
                          else node.pruned_children[a]
                          for a, j in node.children.items()}
            elif self.return_type == 'montecarlo':
                target = {a: self.nodes[j].montecarlo_return
                          if a not in node.pruned_children
                          else (lambda a,b:a/b)(*node.pruned_children[a])
                          for a, j in node.children.items()}

            la_mask = torch.zeros((self.num_actions,))
            la_mask[la] = 1
            data.append((node.id, node.mol.copy(no_blocks=True), la_mask, target, node.reward))
        return data

    def update_values(self):
        if self.return_type == 'max_desc_r':
            self.update_values_max_desc()
        elif self.return_type == 'montecarlo':
            self.update_values_montecarlo()

    def update_values_max_desc(self):
        self.nodes_lock.acquire() # just let whatever's modifying the nodes finish
        self.nodes_lock.release()
        def f(node):
            if node.n_legal_acts == 0:
                node.max_descendant_r = node.reward
                if self.update_prio_on_refresh:
                    self.sumtree.set(node.id, 0)
                return node.reward
            max_r = node.reward
            for i, n in node.children.items():
                if i in node.pruned_children:
                    continue
                n = self.nodes[n]
                max_r = max(max_r, f(n))
            node.max_descendant_r = max_r
            if self.update_prio_on_refresh and node.n_legal_acts > len(node.children):
                self.sumtree.set(node.id, self.score_fn(node))
            return max_r
        print('update values')
        t0 = time.time()
        v0 = f(self.root)
        t1 = time.time()
        print(f'update values took {t1-t0:.3f}s, {v0:.3f}')


    def update_values_montecarlo(self, gamma=1):
        self.nodes_lock.acquire() # just let whatever's modifying the nodes finish
        self.nodes_lock.release()
        def f(node):
            if node.n_legal_acts == 0 or not len(node.children): # This node is a leaf
                return node.reward, 1
            total_return = 0
            total_leaves = 0
            for i, n in node.children.items():
                if i in node.pruned_children:
                    g, nl = node.pruned_children[i]
                else:
                    g, nl = f(self.nodes[n])
                total_return += g
                total_leaves += nl

            node.total_return = total_return * gamma - node.reward
            node.total_leaves = total_leaves
            return total_return, total_leaves
        print('update values')
        t0 = time.time()
        v0 = f(self.root)
        t1 = time.time()
        print(f'update values took {t1-t0:.3f}s, {v0[0]/v0[1]:.3f}')

    def update_v_at(self, idxs, vs):
        self.nodes_lock.acquire() # just let whatever's modifying the nodes finish
        self.nodes_lock.release()
        for j in range(len(idxs)):
            if idxs[j] not in self.nodes: continue
            node = self.nodes[idxs[j]]
            node.value = float(vs[j].copy())
            self.sumtree.set(idxs[j], self.score_fn(node))

    def update_r_at(self, idxs, rs):
        self.nodes_lock.acquire() # just let whatever's modifying the nodes finish
        self.nodes_lock.release()
        #print(idxs, rs)
        for j in range(len(idxs)):
            if idxs[j] not in self.nodes: continue
            if idxs[j] in self.new_node_queue:
                self.new_node_queue.pop(self.new_node_queue.index(idxs[j]))
            node = self.nodes[idxs[j]]
            node.fast_reward = float(rs[j].copy())
            self.sumtree.set(idxs[j], self.score_fn(node))

    def get_num_stored(self):
        return len(self.nodes)

    def prune_tree(self):
        max_rs = []
        def f(node):
            if node.n_legal_acts == 0:
                max_rs.append(node.reward)
                node.max_descendant_r = node.reward
                return
            max_r = node.reward
            for i, n in node.children.items():
                if i in node.pruned_children:
                    continue
                n = self.nodes[n]
                f(n)
                max_r = max(max_r, n.max_descendant_r)
            max_rs.append(max_r)
            node.max_descendant_r = max_r

        removed_nodes = []
        def remove_subtree(node):
            for i, n in node.children.items():
                if n in self.nodes:
                    c = self.nodes[n]
                    removed_nodes.append(c)
                    self.sumtree.set(n, 0)
                    del self.nodes[n]
                    remove_subtree(c)

        def prune(node, thresh):
            for i, n in node.children.items():
                if i in node.pruned_children:
                    continue
                c = self.nodes[n]
                if c.max_descendant_r <= thresh:
                    if self.return_type == 'max_desc_r':
                        node.pruned_children[i] = c.max_descendant_r
                    elif self.return_type == 'montecarlo':
                        node.pruned_children[i] = (c.total_return, c.total_leaves)
                    del self.nodes[n]
                    removed_nodes.append(c)
                    remove_subtree(c)
                    self.sumtree.set(n, 0)
                else:
                    prune(c, thresh)
        t0 = time.time()
        f(self.root) # compute max_descendant_r
        t1 = time.time()
        # compute kth/N percentile value
        kth = find_kth(max_rs, int(len(max_rs) * self.prune_factor))
        t2 = time.time()
        l0 = len(self.nodes)
        self.nodes_lock.acquire()
        gc.collect()
        process = psutil.Process(os.getpid())
        print('mem pre', process.memory_info().rss // (1024 * 1024),
              self.num_explored_nodes, len(self.nodes))
        prune(self.root, kth) # prune nodes below median/percentile value
        self.nodes_lock.release()
        t3 = time.time()
        print(f'prune tree took {t3-t0:.3f}s, ({t1-t0:.3f}s {t2-t1:.3f}s {t3-t2:.3f}s)')
        print(f'{int(self.prune_factor*100)}th percentile = {kth:.3f} reward, {len(self.nodes)} nodes left ({len(max_rs)} - {len(removed_nodes)} = {l0})')
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        t0 = time.time()
        with gzip.open(f'{self.exp_path}/pruned_nodes_{self.num_explored_nodes}.pkl.gz', 'wb') as f:
            pickle.dump(removed_nodes, f, 4)
        t1 = time.time()
        print(f"Saving pruned nodes took {t1-t0:.3f}s")
        del removed_nodes
        gc.collect()
        process = psutil.Process(os.getpid())
        print('mem post', process.memory_info().rss // (1024 * 1024),
              self.num_explored_nodes, len(self.nodes))

    def get_top_k_nodes(self, k, pred_dock=False, sim_dock=False):
        t0 = time.time()
        top_k = [(-10, None, None)]
        self.nodes_lock.acquire()
        for i, n in self.nodes.items():
            if i == 0:
                continue
            if pred_dock and n.pred_dock_reward is not None and n.qed is not None:
                continue
            if sim_dock and n.sim_dock_reward is not None:
                continue
            r = n.reward
            if r > top_k[0][0]:
                top_k.append((r, i, n.mol))
                top_k = sorted(top_k, key=lambda x:x[0])[-k:]
        self.nodes_lock.release() # Todo code version without lock
        t1 = time.time()
        if t1-t0 > 0.1:
            print(f'get_top_k_nodes took {t1-t0:.2f}s', k, sim_dock, pred_dock)
        return top_k

    def set_sim_dock_reward(self, idx, rewards):
        for i, r in zip(idx, rewards):
            if i in self.nodes:
                self.nodes[i].sim_dock_reward = float(r)
                self.sumtree.set(i, self.score_fn(self.nodes[i]))

    def set_pred_dock_reward(self, idx, rewards, qeds, synths):
        for i, r, qed, synth in zip(idx, rewards, qeds, synths):
            if i in self.nodes:
                self.nodes[i].pred_dock_reward = float(r)
                self.nodes[i].qed = qed
                self.nodes[i].synth = synth
                self.sumtree.set(i, self.score_fn(self.nodes[i]))

    def is_full(self):
        return self.num_explored_nodes >= self.max_size - 1

    def take_actions(self, idxs, actions, do_restart=False, return_rejects=False):
        new_idxs = []
        rejects = []
        for i, a in zip(idxs, actions):
            if i in self.nodes:
                new_idx = self.take_action(i, a)
            else:
                new_idx = None
            if new_idx is None and do_restart:
                new_idx = self.sample_many(1, just_idxs=True)[0]
            new_idxs.append(new_idx)

        if do_restart:
            aas = []
            for i in new_idxs:
                node = self.nodes[i]
                # available actions
                aas.append(np.int32([i for i in self.get_legal_actions(node.mol)
                                     if i not in node.children.keys()]))
            if return_rejects:
                return new_idxs, aas, rejects
            return new_idxs, aas

    def take_action(self, idx, action, precomputed=None):
        """takes `action` at node `idx`, if the child node is not terminal, returns its index"""
        if self.num_explored_nodes >= self.max_size - 1:
            if self.num_explored_nodes == self.max_size - 1:
                print('Warning: adding more nodes than capacity')
                self.num_explored_nodes += 1
            return None
        stem_idx = action // self.num_blocks
        block_idx = action % self.num_blocks
        if idx not in self.nodes: # Node probably got pruned
            return None

        node = self.nodes[idx]
        new_mol = self.bdata.add_block_to(node.mol, block_idx, stem_idx)
        legal_actions = self.get_legal_actions(new_mol)
        new_node = Node(0, new_mol, len(legal_actions), idx)
        if precomputed is not None:
            (new_node.fast_reward,
             new_node.pred_dock_reward, new_node.qed,
             new_node.sim_dock_reward) = precomputed

        self.nodes_lock.acquire()
        if (idx not in self.nodes # Node probably got pruned
            or action in node.children # Double sampling?
            ):
            #print("Not inserting", idx, action, idx in self.nodes, action in node.children)
            self.nodes_lock.release()
            return None

        new_idx = self.num_explored_nodes
        # num_explored_nodes may have changed since the first check,
        # but now we have the lock
        if new_idx >= self.max_size - 1:
            if self.num_explored_nodes == self.max_size - 1:
                print('Warning: adding more nodes than capacity')
            self.nodes_lock.release()
            return None
        new_node.id = new_idx
        self.sumtree.set(new_idx, 1) # Normally the RL agent updates
                                     # this later, except for the
                                     # first minibatch

        self.refresh(new_idx)

        self.nodes[new_idx] = new_node
        self.new_node_queue.append(new_idx)
        self.num_explored_nodes += 1
        self.nodes_lock.release()

        node.children[action] = new_idx
        if action in node.pruned_children:   # this node was pruned in the past
            del node.pruned_children[action] # but we're trying it again
        if not len(legal_actions):
            return None
        return new_idx

    def get_legal_actions(self, molecule=None, idx=None, return_mask=False):
        if idx is not None:
            molecule = self.nodes[idx].mol
        atoms_mask = self.bdata.block_natm <= (self.max_atoms - molecule.slices[-1])
        branches_mask = self.bdata.block_nrs <= self.max_branches - len(molecule.stems) - 1
        add_mask = np.logical_and(np.logical_and(atoms_mask, branches_mask),
                                  len(molecule.jbond_atmidxs) < self.max_blocks-1)
        add_mask = np.tile(add_mask[None, :], [self.max_branches, 1])
        num_stems = max(len(molecule.stems), 1 if molecule.numblocks == 0 else 0)
        add_mask[num_stems:, :] = False # Could shrink array instead of padding with False
        add_mask = add_mask.reshape(-1)
        if return_mask:
            return add_mask
        return np.arange(len(add_mask))[add_mask]

    def refresh(self, new_idx):
        if not new_idx % 1000:
            gc.collect()
            process = psutil.Process(os.getpid())
            print('mem', process.memory_info().rss // (1024 * 1024),
                  self.num_explored_nodes, len(self.nodes))

        if not new_idx % 10000:
            new_tree = SumTree(self.max_size)
            for i in range(new_idx):
                new_tree.set(i, self.sumtree.get(i))
            self.sumtree = new_tree


    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        with gzip.open(f'{path}/final_tree.pkl.gz', 'wb') as f:
            pickle.dump(self.nodes, f, 4)

@ray.remote(num_gpus=0.1)
class PredDockRewardActor(PredDockReward):

    def __init__(self, *a, **k):
        synth_config = k.pop('synth_config')
        self.synth_cutoff = k.pop('synth_cutoff')
        super().__init__(*a, **k)
        self.running = True
        self.complete = Complete()
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

    def run(self, tree, n):
        while self.running:
            self.do_iterations(tree, n)

    def stop(self):
        self.running = False

    def do_iterations(self, tree, n):
        mols = ray.get(tree.get_top_k_nodes.remote(n, pred_dock=True))
        if not len(mols):
            time.sleep(1)
            return
        rewards = []
        qeds = []
        synths = []
        idxs = []
        #t0 = time.time()
        for _, i, mol_ in mols:
            if mol_ is None: continue
            mol = mol_.mol
            if mol.GetNumAtoms() == 1:
                rewards.append(-0.5); idxs.append(i); qeds.append((0, 0))
                continue
            atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol, ifcoord=False)
            graph = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
            graph = self.complete(graph)
            batch = Batch.from_data_list([graph]).to(self.device)
            with torch.no_grad():
                pred = self.net(batch)
            qed = QED.qed(mol)
            qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
            qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1

            # Synthesizability constraint
            synth = self.synth_net(mol=mol)
            synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
            synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1

            r = -pred.detach().cpu().numpy().item()
            rewards.append(r)
            idxs.append(i)
            qeds.append((qed, qed_discount))
            synths.append((synth, synth_discount))
        tree.set_pred_dock_reward.remote(idxs, rewards, qeds, synths)
        #t1 = time.time()
        #print(f"Ran {n} docking preds in {t1-t0:.2f}s ({(t1-t0)/n:.2f}s/mol)")
        return

@ray.remote
class SimDockRewardActor:

    def __init__(self, tmp_dir, programs_dir, datasets_dir, num_threads=1):
        self.dock = chem.Dock_smi(tmp_dir,
                                  osp.join(programs_dir, 'chimera'),
                                  osp.join(programs_dir, 'dock6'),
                                  osp.join(datasets_dir, 'brutal_dock/d4/docksetup/'))
        self.target_norm = [-26.3, 12.3]
        self.running = True
        self.num_threads = num_threads

    def run(self, tree, n):
        while self.running:
            self.do_iterations(tree, n)

    def stop(self):
        self.running = False

    def do_iterations(self, tree, n):
        mols = ray.get(tree.get_top_k_nodes.remote(n, sim_dock=True))
        # mols is a list of BlockMoleculeData objects
        mols = [i for i in mols if i[2] is not None]
        n = len(mols)
        rewards = [None] * n
        idxs = [None] * n
        def f(i, idx):
            try:
                _, r, _ = self.dock.dock(Chem.MolToSmiles(mols[i][2].mol))
            except Exception as e: # Sometimes the prediction fails
                print(e)
                r = 0
            rewards[i] = -(r-self.target_norm[0])/self.target_norm[1]
            idxs[i] = idx
        t0 = time.time()
        threads = []
        thread_at = self.num_threads
        for i, (_, idx, mol) in enumerate(mols):
            threads.append(threading.Thread(target=f, args=(i, idx)))
            if i < self.num_threads:
                threads[-1].start()
        while None in rewards:
            if sum([i.is_alive() for i in threads]) < self.num_threads and thread_at < n:
                threads[thread_at].start()
                thread_at += 1
            time.sleep(0.5)
        t1 = time.time()
        print(f"Ran {n} docking simulations in {t1-t0:.2f}s ({(t1-t0)/n:.2f}s/mol)")
        tree.set_sim_dock_reward.remote(idxs, rewards)

@ray.remote(num_gpus=0.1)
class RLActor:
    def __init__(self, model, obs_config, device, train_batcher,
                 act_batcher, tree, temperature, priority_pred):
        self.model = model(obs_config)
        self.model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters(), 1e-3, weight_decay=1e-5)
        self.device = device
        self.train_batcher = train_batcher
        self.act_batcher = act_batcher
        self.tree = tree
        self.temperature = temperature
        self.priority_pred = priority_pred
        # This is a temporary hack because of some annoying warnings
        if 0:
            import os
            import sys
            f = open(os.devnull, 'w')
            sys.stdout = f

    def get_parameters(self):
        return self.model.state_dict()

    def set_parameters_from(self, rl_actor):
        self.model.load_state_dict(ray.get(rl_actor.get_parameters.remote()))

    def set_parameters(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train(self):
        idx, action_mask, graphs, qsa, qsa_mask, rewards = self.train_batcher.get()
        graphs.to(self.device)
        qsa = qsa.to(self.device)
        qsa_mask = qsa_mask.to(self.device)
        action_mask = action_mask.to(self.device).float()
        rewards = rewards.to(self.device)

        r_pred, qsa_pred = self.model(graphs, action_mask)
        q_err = ((qsa_pred - qsa) * qsa_mask)
        q_loss = torch.min(q_err**2, abs(q_err)).sum() / max(1, qsa_mask.sum())
        r_err = (rewards - r_pred)
        r_loss = torch.min(r_err**2, abs(r_err)).mean()
        (q_loss+r_loss).backward()
        self.opt.step()
        self.opt.zero_grad()
        with torch.no_grad():
            if self.priority_pred == 'boltzmann':
                v = (torch.softmax(qsa_pred / self.temperature, 1) * qsa_pred).sum(1)
            elif self.priority_pred == 'greedy_q':
                v = qsa_pred.max(1).values
        if self.priority_pred in ['boltzmann', 'greedy_q']:
            self.tree.update_v_at.remote(idx, v.data.cpu().numpy())
        return q_loss.item(), r_loss.item()


    def get_pol(self, idxs, update_r=True):
        with torch.no_grad():
            idxs, action_mask, graphs = ray.get(self.act_batcher.get_from.remote(idxs))
            graphs.to(self.device)
            action_mask = action_mask.to(self.device).float()
            r_pred, qsa_pred = self.model(graphs, action_mask)
            if update_r:
                self.tree.update_r_at.remote(idxs, r_pred.data.cpu().numpy())
                if self.priority_pred == 'boltzmann':
                    v = (torch.softmax(qsa_pred / self.temperature, 1) * qsa_pred).sum(1)
                elif self.priority_pred == 'greedy_q':
                    v = qsa_pred.max(1).values
                if self.priority_pred in ['boltzmann', 'greedy_q']:
                    self.tree.update_v_at.remote(idxs, v.data.cpu().numpy())
            return torch.softmax(qsa_pred / self.temperature, 1).cpu().data.numpy()




@ray.remote
class MBPrep:
    def __init__(self, env_config, tree, mbsize):
        self.bdata = BlocksData(env_config)
        self.tree = tree
        self.mbsize = mbsize
        self.num_actions = env_config['max_branches'] * env_config['num_blocks']
        self.env_config = env_config

    def compute_graphs(self, mols):
        graphs = []
        for mol in mols:
            mol.restore_blocks(self.bdata)
            atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False,
                                                        one_hot_atom=True, donor_features=False)
            g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
            if self.env_config.get('graph_add_stem_mask', False):
                stem_idx = mol.stem_atmidxs
                stem_mask = torch.zeros((g.x.shape[0], 1))
                stem_mask[torch.tensor(stem_idx).long()] = 1
                g.stem_idx = torch.tensor(stem_idx).long()
                g.x = torch.cat([g.x, stem_mask], 1)
            if g.edge_index.shape[0] == 0:
                g.edge_index = torch.zeros((2, 1)).long()
                g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).float()
                g.stem_idx = torch.zeros((1,)).long()
            graphs.append(g)
        return graphs

    def get_from(self, idx):
        mols, legal_action_mask = ray.get(self.tree.get_mols.remote(idx, return_la_mask=True))
        graphs = self.compute_graphs(mols)
        return idx, legal_action_mask, Batch.from_data_list(graphs)

    def get(self):
        idx, mols, legal_action_mask, qsa_dicts, rewards = zip(
            *ray.get(self.tree.sample_many.remote(self.mbsize)))
        graphs = self.compute_graphs(mols)
        qsa = torch.zeros((len(qsa_dicts), self.num_actions))
        mask = torch.zeros((len(qsa_dicts), self.num_actions))
        for i in range(len(qsa)):
            idxs = torch.tensor(list(qsa_dicts[i].keys())).long()
            qsa[i, idxs] = torch.tensor(list(qsa_dicts[i].values())).float()
            mask[i, idxs] = 1
        return (idx, torch.stack(legal_action_mask), Batch.from_data_list(graphs),
                qsa, mask, torch.tensor(rewards).float())


@ray.remote(num_gpus=0.1)
class RandomRLActor:
    def __init__(self, model, obs_config, device, train_batcher,
                 act_batcher, tree, temperature, priority_pred):
        self.nact = obs_config['num_blocks'] * obs_config['max_branches']

    def get_parameters(self):
        pass

    def set_parameters_from(self, rl_actor):
        pass

    def set_parameters(self, state_dict):
        pass

    def train(self):
        return 0, 0


    def get_pol(self, idxs, update_r=True):
        return np.ones((len(idxs), self.nact)) / self.nact
