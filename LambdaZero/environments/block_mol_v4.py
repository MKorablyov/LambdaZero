import numpy as np
import os.path as osp
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

import LambdaZero.chem
from .molMDP import MolMDP
from rdkit import Chem

import LambdaZero.utils
from .reward import PredDockReward


class FPObs_v1:
    def __init__(self, config, molMDP):
        self.num_blocks = config["num_blocks"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.max_steps = config["max_steps"]
        self.molMDP = molMDP

        num_actions = self.max_blocks + self.max_branches * self.num_blocks

        mol_fp_len = config["obs_config"]["mol_fp_len"] * len(config["obs_config"]["mol_fp_radiis"])
        stem_fp_len = config["obs_config"]["stem_fp_len"] * len(config["obs_config"]["stem_fp_radiis"])

        self.action_space = Discrete(num_actions,)
        self.observation_space = Dict({
            "mol_fp": Box(low=0, high=1, shape=(mol_fp_len,)),
            "stem_fps": Box(low=0, high=1, shape=(self.max_branches, stem_fp_len,)),
            "jbond_fps": Box(low=0, high=1, shape=(self.max_blocks-1, stem_fp_len,)),
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })
        self.get_fps = LambdaZero.chem.FPEmbedding_v2(**config["obs_config"])

    def __call__(self, molecule, num_steps):
        mol_fp, stem_fps_, jbond_fps_ = self.get_fps(molecule)
        # pad indices
        stem_fps = np.zeros([self.max_branches, stem_fps_.shape[1]], dtype=np.float32)
        stem_fps[:stem_fps_.shape[0], :] = stem_fps_[:self.max_branches, :]
        jbond_fps = np.zeros([self.max_blocks - 1, stem_fps_.shape[1]], dtype=np.float32)
        jbond_fps[:jbond_fps_.shape[0], :] = jbond_fps_[:self.max_blocks-1, :]

        # make action mask
        break_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        break_mask[:jbond_fps_.shape[0]] = 1

        # max number of atoms
        atoms_mask = self.molMDP.block_natm <= (self.max_atoms - molecule.slices[-1])
        branches_mask = self.molMDP.block_nrs <= self.max_branches - len(molecule.stems) - 1
        if len(molecule.jbond_atmidxs) == self.max_blocks-1:
            jbonds_mask = np.zeros(self.num_blocks, dtype=np.bool)
        else:
            jbonds_mask = np.ones(self.num_blocks, dtype=np.bool)
        add_mask = np.logical_and(np.logical_and(atoms_mask, branches_mask), jbonds_mask)
        add_mask = np.asarray(add_mask, dtype=np.float32)
        add_mask = np.tile(add_mask[None, :], [self.max_branches, 1])
        num_stems = max(stem_fps_.shape[0], 1 if molecule.numblocks == 0 else 0)
        add_mask[num_stems:, :] = False
        add_mask = add_mask.reshape(-1)
        action_mask = np.concatenate([np.ones([1], dtype=np.float32), break_mask, add_mask])

        obs = {
            "mol_fp": mol_fp,
            "stem_fps": stem_fps,
            "jbond_fps": jbond_fps,
            "action_mask": action_mask,
            "num_steps": num_steps
               }
        return obs

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = {
    "obs_config": {"mol_fp_len": 512,
                  "mol_fp_radiis": [3],
                  "stem_fp_len": 64,
                  "stem_fp_radiis": [4, 3, 2]
                  },

    "molMDP_config": {
        "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
    },

    "reward_config" : {
        "soft_stop": True,
        "load_model": osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
        "natm_cutoff": [45, 50],
        "qed_cutoff": [0.2, 0.7],
        "exp": None,
        "delta": False,
        "simulation_cost": 0.00,
        "device": "cuda",
    },
    "reward": PredDockReward,
    "num_blocks": 105,
    "max_steps": 7,
    "max_blocks": 7,
    "max_atoms": 50,
    "max_branches": 20,
    "random_blocks": 2,
    "max_simulations": 1,
    "allow_removal": True
}


class BlockMolEnv_v4:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems", "blockidxs"]

    def __init__(self, config):

        config = merge_dicts(DEFAULT_CONFIG, config)

        self.num_blocks = config["num_blocks"]
        self.max_blocks = config["max_blocks"]
        self.max_steps = config["max_steps"]
        self.max_simulations = config["max_simulations"]
        self.random_blocks = config["random_blocks"]
        #
        self.molMDP = MolMDP(**config["molMDP_config"])
        self.observ = FPObs_v1(config, self.molMDP)
        self.reward = config["reward"](**config["reward_config"])

        self.action_space = self.observ.action_space
        self.observation_space = self.observ.observation_space

    def _if_terminate(self):
        terminate = False
        # max steps
        if self.num_steps >= self.max_steps:
            terminate = True
        # max simulations
        if self.max_simulations is not None:
            if self.num_simulations >= self.max_simulations:
                terminate = True
        return terminate

    def reset(self):
        self.num_steps = 0
        self.num_simulations = 0
        self.molMDP.reset()
        self.molMDP.random_walk(self.random_blocks)
        self.reward.reset()
        return self.observ(self.molMDP.molecule, self.num_steps)

    def step(self, action):
        if (action == 0):
            simulate = True
            self.num_simulations += 1
        elif action <= (self.max_blocks - 1):
            simulate = False
            self.molMDP.remove_jbond(jbond_idx=action-1)
        else:
            simulate = False
            stem_idx = (action - self.max_blocks)//self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

        self.num_steps += 1
        obs = self.observ(self.molMDP.molecule, self.num_steps)
        done = self._if_terminate()
        reward, info = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        info["molecule"] = self.molMDP.molecule
        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.molMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = deepcopy(self.num_steps)
        num_simulations = deepcopy(self.num_simulations)
        previous_reward = deepcopy(self.reward.previous_reward)
        mol = deepcopy(self.molMDP.molecule._mol)
        return mol_attr, num_steps, num_simulations, previous_reward, mol

    def set_state(self,state):
        mol_attr, self.num_steps, self.num_simulations, self.reward.previous_reward, self.molMDP.molecule._mol \
            = deepcopy(state)
        [setattr(self.molMDP.molecule, key, value) for key, value in mol_attr.items()]
        self.molMDP.molecule._mol = None
        self.molMDP.molecule.blocks = [self.molMDP.block_mols[i] for i in self.molMDP.molecule.blockidxs]
        return self.observ(self.molMDP.molecule, self.num_steps)

    def render(self, outpath):
        mol = self.molMDP.molecule.mol
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)
