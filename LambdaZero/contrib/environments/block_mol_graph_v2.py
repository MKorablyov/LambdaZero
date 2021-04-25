import os.path as osp
import numpy as np
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

from LambdaZero.environments.molMDP import MolMDP
import LambdaZero.utils
from LambdaZero.environments.block_mol_v3 import synth_config
from LambdaZero.contrib.reward import DummyReward
from LambdaZero.environments.block_mol_graph_v1 import GraphMolObs

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = {
    "obs_config": {
    },
    "molMDP_config": {
        "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
    },
    "reward_config": {
        "binding_model": osp.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth"),
        "qed_cutoff": [0.2, 0.5],
        "synth_cutoff": [0, 4],
        "synth_config": synth_config,
        "device": "cuda",
    },
    "reward": DummyReward,
    "num_blocks": None, # 105, 464
    "max_steps": 7,
    "max_blocks": 15,
    "max_atoms": 55,
    "max_branches": 50,
    "random_steps": 5,
    "reset_min_blocks": 3,
    "allow_removal": True
}


class BlockMolGraph_v2:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]
    def __init__(self, config=None):
        config = merge_dicts(DEFAULT_CONFIG, config)

        self.molMDP = MolMDP(**config["molMDP_config"])
        if config["num_blocks"] is None:
            self.num_blocks = len(self.molMDP.block_smi)
        else:
            self.num_blocks = config["num_blocks"]

        self.max_steps = config["max_steps"]
        self.max_branches = config["max_branches"]
        self.max_blocks = config["max_blocks"]
        self.max_atoms = config["max_atoms"]
        self.random_steps = config["random_steps"]
        self.reset_min_blocks = config["reset_min_blocks"]
        self.allow_removal = config["allow_removal"]
        num_actions = self.max_blocks + self.max_branches * self.num_blocks

        self.graph_mol_obs = GraphMolObs(config['obs_config'], self.max_branches, self.max_blocks - 1)

        self.action_space = Discrete(num_actions, )
        self.action_space.max_branches = self.max_branches
        self.action_space.max_blocks = self.max_blocks
        self.action_space.num_blocks = self.num_blocks

        self.observation_space = Dict({
            "mol_graph": self.graph_mol_obs.space,
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })
        self.reward = config["reward"](**config["reward_config"])

    def _action_mask(self, verbose=False):
        # break actions
        jbond_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        if self.allow_removal:
            jbond_mask[:len(self.molMDP.molecule.jbonds)] = 1
        # add actions
        stem_mask = np.zeros(self.max_branches, dtype=np.float32)
        if self.molMDP.molecule.numblocks == 0:
            stem_mask[0] = 1 # allow to add any block
        else:
            stem_mask[:len(self.molMDP.molecule.stems)] = 1
        stem_mask = np.tile(stem_mask[:, None], [1, self.num_blocks]).reshape(-1)
        # all actions
        action_mask = np.concatenate([np.ones([1], dtype=np.float32), jbond_mask, stem_mask])
        return action_mask

    def _make_obs(self):
        action_mask = self._action_mask()
        graph, flat_graph = self.graph_mol_obs(self.molMDP.molecule)
        obs = {"mol_graph": flat_graph,
               "action_mask": action_mask,
               "num_steps": self.num_steps}
        return obs, None

    def _should_stop(self):
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

    def _random_steps(self):
        for i in range(self.random_steps):
            self.num_steps = 0
            self.molMDP.reset()
            self.reward.reset()
            obs, graph = self._make_obs()

            for i in range(self.random_steps):
                actions = np.where(obs["action_mask"])[0]
                action = np.random.choice(actions)
                self.step(action)
                if self._should_stop(): self.molMDP.reset()
                obs, graph = self._make_obs()
                self.num_steps = 0
            return obs

    def reset(self):
        feasible_state = False
        while not feasible_state:
            obs = self._random_steps()
            try:
                assert self.molMDP.molecule is not None, "molecule is None"
                # try if the molecule produces valid smiles string
                self.molMDP.molecule.smiles
            except Exception as e:
                continue
            if self.molMDP.molecule.numblocks < self.reset_min_blocks:
                continue
            feasible_state = True
        return obs

    def step(self, action):
        self.num_steps += 1
        # check if action is legal
        action_mask = self._action_mask()
        if not action in np.where(action_mask)[0]:
            raise ValueError('illegal action:', action, "available", np.sum(action_mask))

        if (action == 0): # 0
            agent_stop = True
        elif action <= (self.max_blocks - 1): # 3 max_blocks; action<=2; 0,1,2
            agent_stop = False
            self.molMDP.remove_jbond(jbond_idx=action-1)
        else:
            agent_stop = False
            stem_idx = (action - self.max_blocks) // self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

        obs, graph = self._make_obs()
        self.molMDP.molecule.graph = graph
        # if env should stop
        env_stop = self._should_stop()
        done = agent_stop or env_stop
        # compute reward
        reward, log_vals = self.reward(self.molMDP.molecule, agent_stop, env_stop, self.num_steps)
        return obs, reward, done, {"log_vals":log_vals}