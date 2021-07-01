import copy
import os.path as osp
import time

import numpy as np
from gym.spaces import Space, Discrete, Box, Dict
from ray.rllib.utils import merge_dicts
import torch
from torch_geometric.data import Data, Batch
import zlib
import struct
import traceback

import LambdaZero.chem as chem
import LambdaZero.utils

from lightrl.env.molMDP import MolMDP
from lightrl.env.block_mol_v3 import DEFAULT_CONFIG as block_mol_v3_config, BlockMolEnv_v3, REWARD_CLASS
from rdkit import Chem
from rdkit.Chem import QED
from chemprop.features import BatchMolGraph, MolGraph


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


DEFAULT_CONFIG = dict(block_mol_v3_config)
DEFAULT_CONFIG.update({
    "obs_config": {'one_hot_atom': False},
    "threshold": 0.7
})


class MolGraphSpace(Space):
    def __init__(self, num_node_feat=1, num_edge_feat=1,
                 attributes=['x', 'edge_index', 'edge_attr']):
        self.num_node_feat = num_node_feat
        self.num_edge_feat = num_edge_feat
        self.attributes = attributes
        self._fg = False
        self._size = 8096 * 2
        self.shape = (self._size,)
        self._ndims = None
        self._shapeslen = None
        self._dtypes = None

    def contains(self, x):
        return True

    def pack(self, g):
        msg = b''
        shapes = np.int16(np.hstack([getattr(g, i).shape for i in self.attributes])).tostring()
        if self._ndims is None:
            self._ndims = [getattr(g, i).ndim for i in self.attributes]
            self._shapeslen = len(shapes)
            self._dtypes = [getattr(g, i).numpy().dtype for i in self.attributes]
        msg += shapes
        msg += b''.join([getattr(g, i).numpy().data for i in self.attributes])
        msg = zlib.compress(msg)
        buf = np.zeros(self._size, dtype=np.uint8)
        buf[:2] = np.frombuffer(np.uint16(len(msg)).data, np.uint8)
        buf[2:2+len(msg)] = np.frombuffer(msg, np.uint8)
        return buf

    def unpack(self, buf):
        l, = struct.unpack('H', buf[:2])
        msg = zlib.decompress(buf[2:2+l])
        d = {}
        shapes = struct.unpack((self._shapeslen // 2) * 'h', msg[:self._shapeslen])
        idx = self._shapeslen
        for attr, ndim, dtype, dsize in zip(self.attributes, self._ndims, self._dtypes, self._dsizes):
            shape = shapes[:ndim]
            l = shape[0] * (1 if ndim == 1 else shape[1]) * dsize

            arr = np.copy(np.frombuffer(msg[idx:idx+l], dtype).reshape(shape)) # copy to make array writeable
            d[attr] = torch.from_numpy(arr)
            idx += l
            shapes = shapes[ndim:]
        return ParametricMolData(**d)

    def to_jsonable(self, batch):
        return {attr: [getattr(i, attr).numpy().tolist() for i in batch]
                for attr in self.attributes}


    def from_jsonable(self, batch):
        return [ParametricMolData(**{attr: torch.tensor(i)
                                     for attr, i in zip(self.attributes, data)})
                for data in zip(*[batch[i] for i in self.attributes])]

class ParametricMolData(Data):
    def __inc__(self, key, value):
        if key == 'stem_atmidx' or key == 'jbond_atmidx':
            return self.num_nodes
        return super().__inc__(key, value)


class GraphMolObs:
    def __init__(self, config={}, max_stems=25, max_jbonds=10):
        self.one_hot_atom = config.get('one_hot_atom', False)
        self.stem_indices = config.get('stem_indices', True)
        self.jbond_indices = config.get('jbond_indices', True)
        self.max_stems = max_stems
        self.max_jbonds = max_jbonds
        num_feat = 14
        num_feat += len(chem.atomic_numbers) if self.one_hot_atom else 0
        self.num_base_feat = num_feat
        #num_feat += 1 if self.stem_indices else 0
        #num_feat += 1 if self.jbond_indices else 0

        self.space = MolGraphSpace(num_node_feat=num_feat,
                                   num_edge_feat=4,
                                   attributes=['x', 'edge_index', 'edge_attr',
                                               'stem_atmidx', 'jbond_atmidx',
                                               # 'stem_preds', 'jbond_preds'
                                               ])
        self.space._ndims = [2,2,2,1,2,2,2]
        self.space._shapeslen = 26
        self.space._dtypes = [np.float32, np.int64, np.float32, np.int64,
                              np.int64, np.float32, np.float32]
        self.space._dsizes = [i().itemsize for i in self.space._dtypes]

    def __call__(self, mol, flatten=False):
        # I'm giving up on trying to understand the RLLib internals
        # well enough to have it incorporate the variable length
        # graphs encoded into the data/batch buffer idk that RLLib
        # uses to handle observation data. This is really
        # unnecessary... maybe someone with more courage than me can
        # make this more efficient.

        # Another inefficiency is that RLLib's action space must also
        # be fixed size. This forces a lot of useless computation as
        # here for example I pad the stem_admidx with zeros and simply
        # ignore (mask) the result of the computation later. This is
        # faster than _not_ doing the computation and then zero
        # padding the _results_ to fit the action space length.

        if mol.mol is None:
            atmfeat, bond, bondfeat = (np.zeros((1, self.num_base_feat)), np.zeros((1, 2)),
                                       np.zeros((1, self.space.num_edge_feat)))
        else:
            atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False)
        g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat, data_cls=ParametricMolData)
        # g.stem_preds = torch.zeros((self.max_stems, 0))
        # g.jbond_preds = torch.zeros((self.max_jbonds, 0))

        if self.stem_indices: # Add stem indices
            stem_idx = mol.stem_atmidxs[:self.max_stems]
            stem_mask = torch.zeros((g.x.shape[0], 1))
            stem_mask[torch.tensor(stem_idx).long()] = 1
            g.stem_atmidx = torch.tensor(
                np.concatenate([stem_idx, np.zeros(self.max_stems - len(stem_idx))], 0)).long()
            #g.x = torch.cat([g.x, stem_mask], 1)

        if self.jbond_indices: # Add jbond indices
            jbond_idx = mol.jbond_atmidxs[:self.max_jbonds]
            jbond_mask = torch.zeros((g.x.shape[0], 1))
            jbond_mask[torch.tensor(jbond_idx.flatten()).long()] = 1
            if len(jbond_idx) == 0:
                g.jbond_atmidx = torch.zeros((self.max_jbonds, 2)).long()
            else:
                g.jbond_atmidx = torch.tensor(
                    np.concatenate([jbond_idx,np.zeros((self.max_jbonds - len(jbond_idx), 2))], 0)).long()
            #g.x = torch.cat([g.x, jbond_mask], 1)

        if g.edge_index.shape[0] == 0: # Edge case
            g.edge_index = torch.zeros((2, 1)).long()
            g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).float()
            g.stem_atmidx = torch.zeros((self.max_stems,)).long()

        # todo: check if emty graphs could be acquired!!
        g_flat = None
        if flatten:
            g_flat = self.space.pack(g)
        #g_ = self.space.unpack(g_flat)
        #if g is not None:
            #print("g_:", g_)
        #for var in vars(g):
        #    if getattr(g,var) is not None:
        #        print(torch.equal(getattr(g,var), getattr(g_, var)))
        #        print(torch.sum(getattr(g,var) - getattr(g_, var)))
        #    else:
        #        print(var, "is None!")
        return g, g_flat


class BlockMolEnvGraph_v1(BlockMolEnv_v3):

    def __init__(self, config=dict(), *args, proc_id=0, **kwargs):
        self._config = config = merge_dicts(DEFAULT_CONFIG, config)
        self._proc_id = proc_id
        self._eval_mode = config.get("eval_mode", False)

        self.num_blocks = config["num_blocks"]
        self.molMDP = MolMDP(**config["molMDP_config"])
        if self.num_blocks is None:
            self.num_blocks = len(self.molMDP.block_smi)

        self.max_steps = config["max_steps"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.random_steps = config["random_steps"]
        self.allow_removal = config["allow_removal"]
        num_actions = self.max_blocks + self.max_branches * self.num_blocks

        self.graph_mol_obs = GraphMolObs(config['obs_config'], self.max_branches, self.max_blocks-1)

        self.action_space = Discrete(num_actions,)
        self.action_space.max_branches = self.max_branches
        self.action_space.max_blocks = self.max_blocks
        self.action_space.num_blocks = self.num_blocks
        self.observation_space = Dict({
            # "mol_graph": self.graph_mol_obs.space,
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
            # "chosen_seed": Box(low=0, high=1, shape=(num_actions,)),
        })
        print(f"Using reward class {config['reward']}")
        config["reward"] = REWARD_CLASS[config["reward"]]

        self.reward = config["reward"](**config["reward_config"])

        self.reward = config["reward"](**config["reward_config"])
        self._prev_obs = None
        self._mock_obs = None

        worker_index = proc_id  # getattr(config, "worker_index", 0)
        vector_index = getattr(config, "vector_index", 0)
        self._pre_saved_graph = getattr(config, "pre_saved_graph", False)
        self._reset_mode = False
        self._env_rnd_state = None
        self._env_seed = None

        self._action_list = np.arange(self.max_blocks + self.max_branches * self.num_blocks)

        self._proc_rnd = np.random.RandomState(proc_id)
        self._chosen_seed = -1

        self.set_seed(config, worker_index=worker_index, vector_index=vector_index)

        if self._pre_saved_graph:
            self._saved_graphs = torch.load(f"{datasets_dir}/mol_data.pkl")

        self.obs_cuda = config.get("obs_cuda", True)

    def _debug_step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            with open(osp.join(summaries_dir, 'block_mol_graph_v1.error.txt'), 'a') as f:
                print(e, file=f)
                print(self.get_state(), file=f)
                print(action, file=f)
                print(traceback.format_exc(), file=f)
                print(traceback.format_exc(), file=f)
                f.flush()
            print(e)
            print(self.get_state())
            print(action)
            print(traceback.format_exc())
            super().reset()  # reset the environment
            return super().step(0)  # terminate

    def _make_obs(self, flat=False):
        # if self._mock_obs is not None:
        #     return self._mock_obs

        mol = self.molMDP.molecule

        if self._pre_saved_graph:
            graph = self._saved_graphs[mol.smiles]
            flat_graph = None
        else:
            graph, flat_graph = self.graph_mol_obs(mol, flatten=flat)  # TODO SLOW

        # TODO improve data types
        # # ['x', 'edge_index', 'edge_attr', 'stem_preds', 'jbond_preds', 'stem_atmidx', 'jbond_atmidx']
        # for x in graph.keys:
        #     graph.__dict__[x] = graph.__dict__[x].byte()
        # graph = graph.to(torch.device("cuda"))

        # make action mask
        action_mask = self._get_action_mask()
        # synth_score = 0
        # for x in graph.keys:
        #     graph[x].share_memory_()
        if self.obs_cuda:
            graph = graph.cuda(non_blocking=True)

        obs = {"mol_graph": flat_graph if flat else graph,
               "action_mask": action_mask,
               "rcond": 0,  # TODO should not be here
               "r_steps": self.max_steps - self.num_steps,
               "seed": self._chosen_seed}

        self._prev_obs = obs

        return obs, graph