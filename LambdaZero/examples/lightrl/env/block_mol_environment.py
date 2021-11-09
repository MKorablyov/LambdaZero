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

from LambdaZero.examples.lightrl.env.mol_mdp_ext import MolMDPExtended
from LambdaZero.examples.lightrl.env.block_mol_v3 import DEFAULT_CONFIG as block_mol_v3_config, BlockMolEnv_v3, REWARD_CLASS


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


DEFAULT_CONFIG = dict(block_mol_v3_config)
DEFAULT_CONFIG.update({
    "obs_config": {'one_hot_atom': False},
    "threshold": 0.7
})


class BlockMoleculeEnvironment(BlockMolEnv_v3):

    def __init__(self, config=dict(), *args, proc_id=0, **kwargs):
        self._config = config = merge_dicts(DEFAULT_CONFIG, config)
        self._proc_id = proc_id
        self._eval_mode = config.get("eval_mode", False)


        # ==========================================================================================
        # MDP LOAD
        bpath = osp.join(datasets_dir, getattr(config["mdp_init"], "bpath", "fragdb/blocks_PDB_105.json"))
        self.molMDP = MolMDPExtended(bpath)  # Used for generating representation
        mdp_init = config.get("mdp_init", {"repr_type": "atom_graph"})
        mdp_init = getattr(mdp_init, "__dict__", mdp_init)
        mdp_init["device"] = torch.device("cpu")
        self.molMDP.post_init(**mdp_init)

        self.molMDP.build_translation_table()
        # ==========================================================================================

        self.num_blocks = len(self.molMDP.block_smi)

        self.max_steps = config["max_steps"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.random_steps = config["random_steps"]
        self.allow_removal = config["allow_removal"]
        num_actions = self.max_blocks + self.max_branches * self.num_blocks

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

        worker_index = proc_id
        vector_index = config.get("vector_index", 0)
        self._pre_saved_graph = False  # TODO Fix Deprecated  config.get("pre_saved_graph", False)
        self._reset_mode = False
        self._env_rnd_state = None
        self._env_seed = None

        self._action_list = np.arange(self.max_blocks + self.max_branches * self.num_blocks)

        self._proc_rnd = np.random.RandomState(proc_id)
        self._chosen_seed = -1

        self.set_seed(config, worker_index=worker_index, vector_index=vector_index)

        if self._pre_saved_graph:
            self._saved_graphs = torch.load(f"{datasets_dir}/mol_data.pkl")

        self.obs_cuda = config.get("obs_cuda", torch.cuda.is_available())

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

        if flat:
            raise NotImplementedError

        graph = self.molMDP.mol2repr(mol)  # TODO SLOW

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

        obs = {"mol_graph": graph,
               "action_mask": action_mask,
               "rcond": 0,  # TODO should not be here
               "r_steps": self.max_steps - self.num_steps,
               "seed": self._chosen_seed}

        self._prev_obs = obs

        return obs, graph