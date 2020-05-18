import numpy as np
from rdkit import Chem
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box

import LambdaZero.chem
from .molMDP import MolMDP


class BlockMolEnv_v3:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]
    # sample termination (0 reward if not terminated)

    def __init__(self, config=None):
        self.num_blocks = config["num_blocks"]
        self.max_steps = config["max_steps"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.random_steps = config["random_steps"]
        self.allow_removal = config["allow_removal"]
        num_actions = self.max_blocks + self.max_branches * self.num_blocks

        self.action_space = Discrete(num_actions,)
        mol_fp_len = config["obs_config"]["mol_fp_len"] * len(config["obs_config"]["mol_fp_radiis"])
        stem_fp_len = config["obs_config"]["stem_fp_len"] * len(config["obs_config"]["stem_fp_radiis"])
        self.observation_space = Dict({
            "mol_fp": Box(low=0, high=1, shape=(mol_fp_len,)),
            "stem_fps": Box(low=0, high=1, shape=(self.max_branches, stem_fp_len,)),
            "jbond_fps": Box(low=0, high=1, shape=(self.max_blocks -1, stem_fp_len,)),
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })

        self.molMDP = MolMDP(**config["molMDP_config"])
        self.reward = config["reward"](**config["reward_config"])
        self.get_fps = LambdaZero.chem.FPEmbedding_v2(**config["obs_config"])

    def _make_obs(self):
        mol_fp, stem_fps_, jbond_fps_ = self.get_fps(self.molMDP.molecule)

        # pad indices
        stem_fps = np.zeros([self.max_branches, stem_fps_.shape[1]], dtype=np.float32)
        stem_fps[:stem_fps_.shape[0], :] = stem_fps_[:self.max_branches,:]
        jbond_fps = np.zeros([self.max_blocks -1, stem_fps_.shape[1]], dtype=np.float32)
        jbond_fps[:jbond_fps_.shape[0], :] = jbond_fps_[:self.max_blocks-1, :]

        # make action mask
        jbond_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        if self.allow_removal:
            jbond_mask[:jbond_fps_.shape[0]] = 1
        stem_mask = np.zeros(self.max_branches, dtype=np.float32)
        if self.molMDP.molecule.numblocks == 0: stem_mask[0] = 1 # allow to add any block
        else: stem_mask[:stem_fps_.shape[0]] = 1

        stem_mask = np.tile(stem_mask[:, None], [1, self.num_blocks]).reshape(-1)
        action_mask = np.concatenate([np.ones([1], dtype=np.float32), jbond_mask, stem_mask])

        obs = {"mol_fp": mol_fp,
               "stem_fps": stem_fps,
               "jbond_fps": jbond_fps,
               "action_mask": action_mask,
               "num_steps": self.num_steps}

        return obs

    def _if_terminate(self):
        terminate = False
        molecule = self.molMDP.molecule
        # max steps
        if self.num_steps >= self.max_steps: terminate = True
        # max_branches
        if len(molecule.stems) >= self.max_branches: terminate = True
        # max blocks
        if len(molecule.jbond_atmidxs) >= self.max_blocks-1: terminate = True
        # max_atoms
        if molecule.slices[-1] >= self.max_atoms: terminate = True
        return terminate

    def reset(self):
        self.num_steps = 0
        self.molMDP.reset()
        self.reward.reset()
        obs = self._make_obs()
        for i in range(self.random_steps):
            actions = np.where(obs["action_mask"])[0]
            action = np.random.choice(actions)
            self.step(action)
            obs = self._make_obs()
            if self._if_terminate():
                self.num_steps = 0
                self.molMDP.reset()
        self.num_steps = 0
        return obs

    def step(self, action):
        if (action == 0):
            agent_stop = True
        elif action <= (self.max_blocks - 1):
            agent_stop = False
            self.molMDP.remove_jbond(jbond_idx=action-1)
        else:
            agent_stop = False
            stem_idx = (action - self.max_blocks) // self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

        self.num_steps += 1
        obs = self._make_obs()
        env_stop = self._if_terminate()
        reward, info = self.reward(self.molMDP.molecule, agent_stop, env_stop, self.num_steps)
        info["molecule"] = self.molMDP.molecule
        done = any((agent_stop, env_stop))
        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.molMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = self.num_steps
        return mol_attr, num_steps

    def set_state(self,state):
        mol_attr, self.num_steps = state
        [setattr(self.molMDP.molecule, key, deepcopy(value)) for key, value in mol_attr.items()]
        self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx] for idx in state[0]["blockidxs"]]
        self.molMDP.molecule._mol = None
        return self._make_obs()

    def render(self, outpath):
        mol = self.molMDP.molecule.mol
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)