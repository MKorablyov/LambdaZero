import time
import os.path as osp
import numpy as np
from rdkit import Chem
from copy import deepcopy
from gym.spaces import Space, Discrete, Box, Dict
from ray.rllib.utils import merge_dicts
import torch
from torch_geometric.data import Data, Batch
import pickle
import zlib
import struct
import traceback

import LambdaZero.chem as chem
import LambdaZero.utils
from .molMDP import MolMDP
from .reward import PredDockReward
from .block_mol_v3 import DEFAULT_CONFIG as block_mol_v3_config, BlockMolEnv_v3

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = dict(block_mol_v3_config)
DEFAULT_CONFIG.update({
    "obs_config": {'one_hot_atom': True},
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
            d[attr] = torch.from_numpy(np.frombuffer(msg[idx:idx+l], dtype).reshape(shape))
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

    def __init__(self, config={}, max_stems=20, max_jbonds=10):
        self.one_hot_atom = config.get('one_hot_atom', True)
        self.stem_indices = config.get('stem_indices', True)
        self.jbond_indices = config.get('jbond_indices', True)
        self.max_stems = max_stems
        self.max_jbonds = max_jbonds
        num_feat = 14
        num_feat += len(chem.atomic_numbers) if self.one_hot_atom else 0
        self.num_base_feat = num_feat
        num_feat += 1 if self.stem_indices else 0
        num_feat += 1 if self.jbond_indices else 0

        self.space = MolGraphSpace(num_node_feat=num_feat,
                                   num_edge_feat=4,
                                   attributes=['x', 'edge_index', 'edge_attr',
                                               'stem_atmidx', 'jbond_atmidx',
                                               'stem_preds', 'jbond_preds'])
        self.space._ndims = [2,2,2,1,2,2,2]
        self.space._shapeslen = 26
        self.space._dtypes = [np.float32, np.int64, np.float32, np.int64,
                              np.int64, np.float32, np.float32]
        self.space._dsizes = [i().itemsize for i in self.space._dtypes]

    def __call__(self, mol):
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
            atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol.mol, ifcoord=False,
                                                        one_hot_atom=True, donor_features=False)
        g = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat, data_cls=ParametricMolData)
        g.stem_preds = torch.zeros((self.max_stems, 0))
        g.jbond_preds = torch.zeros((self.max_jbonds, 0))

        if self.stem_indices: # Add stem indices
            stem_idx = mol.stem_atmidxs[:self.max_stems]
            stem_mask = torch.zeros((g.x.shape[0], 1))
            stem_mask[torch.tensor(stem_idx).long()] = 1
            g.stem_atmidx = torch.tensor(
                np.concatenate([stem_idx, np.zeros(self.max_stems - len(stem_idx))], 0)).long()
            g.x = torch.cat([g.x, stem_mask], 1)

        if self.jbond_indices: # Add jbond indices
            jbond_idx = mol.jbond_atmidxs[:self.max_jbonds]
            jbond_mask = torch.zeros((g.x.shape[0], 1))
            jbond_mask[torch.tensor(jbond_idx.flatten()).long()] = 1
            if len(jbond_idx) == 0:
                g.jbond_atmidx = torch.zeros((self.max_jbonds, 2)).long()
            else:
                g.jbond_atmidx = torch.tensor(
                    np.concatenate([jbond_idx,
                                    np.zeros((self.max_jbonds - len(jbond_idx), 2))], 0)).long()
            g.x = torch.cat([g.x, stem_mask], 1)

        if g.edge_index.shape[0] == 0: # Edge case
            g.edge_index = torch.zeros((2, 1)).long()
            g.edge_attr = torch.zeros((1, g.edge_attr.shape[1])).float()
            g.stem_atmidx = torch.zeros((self.max_stems,)).long()

        return self.space.pack(g)



class BlockMolEnvGraph_v1(BlockMolEnv_v3):
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]

    def __init__(self, config=None):

        config = merge_dicts(DEFAULT_CONFIG, config)

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
            "mol_graph": self.graph_mol_obs.space,
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })

        self.reward = config["reward"](**config["reward_config"])
        self._prev_obs = None


    def step(self, action):
        try:
            return super().step(action)
        except Exception as e:
            with open(osp.join(summaries_dir, 'block_mol_graph_v1.error.txt'), 'a') as f:
                print(e, file=f)
                print(self.get_state(), file=f)
                print(action, file=f)
                print(traceback.format_exc(), file=f)
                f.flush()
            print(e)
            print(self.get_state())
            print(action)
            print(traceback.format_exc())
            return super().step(0)

    def _make_obs(self):
        mol = self.molMDP.molecule
        graph = self.graph_mol_obs(mol)

        # make action mask
        jbond_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        if self.allow_removal:
            jbond_mask[:len(mol.jbonds)] = 1
        stem_mask = np.zeros(self.max_branches, dtype=np.float32)
        if self.molMDP.molecule.numblocks == 0: stem_mask[:] = 1 # allow to add any block
        else: stem_mask[:len(mol.stems)] = 1

        stem_mask = np.tile(stem_mask[:, None], [1, self.num_blocks]).reshape(-1)
        action_mask = np.concatenate([np.ones([1], dtype=np.float32), jbond_mask, stem_mask])

        obs = {"mol_graph": graph,
               "action_mask": action_mask,
               "num_steps": self.num_steps}

        self._prev_obs = obs

        return obs
