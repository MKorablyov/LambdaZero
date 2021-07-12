import time
import warnings
import os.path as osp

import gym
import gym.core
import numpy as np
from rdkit import Chem
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

import LambdaZero.chem
import LambdaZero.utils
from LambdaZero.examples.lightrl.env.molMDP import MolMDP
from LambdaZero.examples.lightrl.env.reward import PredockedReward, DummyReward


REWARD_CLASS = {
    "PredockedReward": PredockedReward,
    "DummyReward": DummyReward,
}

import traceback
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

synth_config = {
    "trainer_config": {
        "dataset_type": "regression",
        "train_dataset": None,
        "save_dir": None,

        "features_generator": None,  # Method(s) of generating additional features
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the acquire data loading (0 means sequential)
        "batch_size": 50,  # Batch size
        "separate_val_path": None,  # Path to separate val set, optional
        "separate_test_path": None,  # Path to separate test set, optional
        "split_type": "random",
        "split_sizes": (0.8, 0.1, 0.1),
        "num_folds": 1,
        "seed": 0,
        "pytorch_seed": 0,
        "log_frequency": 10,
        "cache_cutoff": 10000,
        "save_smiles_splits": False,

        "hidden_size": 300,
        "depth": 3,
        "dropout": 0.0,
        "activation": "ReLu",
        "ffn_num_layers": 2,
        "ensemble_size": 1,
        "atom_messages": False,  # Centers messages on atoms instead of on bonds
        "undirected": False,

        "epochs": 150,
        "warmup_epochs": 2.0,  # epochs for which lr increases linearly; afterwards decreases exponentially
        "init_lr": 1e-4,  # Initial learning rate
        "max_lr": 1e-3,  # Maximum learning rate
        "final_lr":  1e-4,  # Final learning rate
        "class_balance": False,
        },
    "predict_config": {
        "dataset_type": "regression",
        "features_generator": None,
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the acquire data loading (0 means sequential)
        "batch_size": 50,  # Batch size
        "disable_progress_bar": True,
        "checkpoint_path": osp.join(datasets_dir, "Synthesizability/MPNN_model/Regression/model_2/model.pt")
    },

}


DEFAULT_CONFIG = {
    "obs_config": {"mol_fp_len": 512,
                  "mol_fp_radiis": [3],
                  "stem_fp_len": 64,
                  "stem_fp_radiis": [4, 3, 2]
                  },
    "molMDP_config": {
        "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
    },

    "reward_config": {
        "qed_cutoff": [0.3, 0.7],
        "synth_cutoff": [0, 4],
        "synth_config": synth_config,
        "soft_stop": True,
        "exp": None,
        "delta": False,
        "simulation_cost": 0.0,
        "device": "cuda",
        "qed_th": 0.3,
        "synth_th": 4.,
    },

    "reward": "DummyReward",
    "num_blocks": 105, # 105, 464
    "max_steps": 7,
    "max_blocks": 7,
    "max_atoms": 50,
    "max_branches": 20,
    "random_steps": 3,
    "allow_removal": True,

    "env_seed": 123,  # int - seed number / list[int] Same init state from seed list
    "eval_mode": False,
    "evaluation_num_episodes": None, # Set from config validation
    "filter_init_states": [],
    "logger": None,
    "random_sample_seed": True,
    "save_eval_data": False,
    "pre_saved_graph": False,
}


class BlockMolEnv_v3(gym.core.Env):
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]

    def __init__(self, config=None, proc_id=0):
        self._config = config = merge_dicts(DEFAULT_CONFIG, config)

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

        print(f"Using reward class {config['reward']}")
        config["reward"] = REWARD_CLASS[config["reward"]]

        self.reward = config["reward"](**config["reward_config"])

        self.get_fps = LambdaZero.chem.FPEmbedding_v2(**config["obs_config"])

        self._prev_obs = None

        worker_index = proc_id
        vector_index = config.get("vector_index", 0)
        self._reset_mode = False
        self._env_rnd_state = None
        self._env_seed = None
        self._eval_mode = False
        self._action_list = np.arange(self.max_blocks + self.max_branches * self.num_blocks)
        self._proc_rnd = np.random.RandomState(proc_id)

        self.set_seed(config, worker_index=worker_index, vector_index=vector_index)
        self._crt_episode = None
        self._chosen_seed = -1

    def set_seed(self, config, worker_index=0, vector_index=0):
        """ Configure env seeds (3 types):
            (1) list of fixed random seeds which can be sampled randomly or sequentially
            (2) 1 seed for each worker set at the beginning
            (3) random seed
        """
        if isinstance(config["env_seed"], list) or isinstance(config["env_seed"], np.ndarray):
            # List
            self._random_sample_seed = config["random_sample_seed"]

            if not self._random_sample_seed:
                self._seed_cnt = 0

            self._env_rnd_state = config["env_seed"]
            print(f"Env seeds: Count: {len(config['env_seed'])} "
                  f"| Starts with: {config['env_seed'][0]}"
                  f"| Ends with: {config['env_seed'][-1]}")

            def ret_rnd_state(seed_id: int = None):
                if seed_id is not None:
                    seed = self._env_rnd_state[seed_id % len(self._env_rnd_state)]
                elif self._random_sample_seed:
                    seed = self._proc_rnd.choice(self._env_rnd_state)
                else:
                    seed = self._env_rnd_state[self._seed_cnt]
                    self._seed_cnt = (self._seed_cnt + 1) % len(self._env_rnd_state)
                    print(f"SEED list:: {seed}")
                self._chosen_seed = seed

                return np.random.RandomState(seed)

            self._env_seed = ret_rnd_state
        elif isinstance(config["env_seed"], int):
            unique_offset = vector_index * 1000 + worker_index
            print(f"USING SEED offset: {unique_offset}")
            self._env_rnd_state = np.random.RandomState(config["env_seed"] + unique_offset)

            def ret_rnd_state(seed_id: int = None):
                return self._env_rnd_state
            self._env_seed = ret_rnd_state
        else:
            rnd_seed = np.random.randint(333333)
            print(f"USING rand seed: {rnd_seed}")
            self._env_rnd_state = np.random.RandomState(rnd_seed)

            def ret_rnd_state(seed_id: int = None):
                return self._env_rnd_state
            self._env_seed = ret_rnd_state

    def _make_obs(self):
        raise NotImplementedError

    def _if_terminate(self):
        terminate = False
        molecule = self.molMDP.molecule
        # max steps
        if self.num_steps >= self.max_steps and not self._reset_mode: terminate = True
        # max_branches
        if len(molecule.stems) >= self.max_branches: terminate = True
        # max blocks
        if len(molecule.jbond_atmidxs) >= self.max_blocks-1: terminate = True
        # max_atoms
        if molecule.slices[-1] >= self.max_atoms: terminate = True
        return terminate

    def reset(self, episode: int=None, **kwargs):
        self._crt_episode = episode

        self._reset_mode = True  # Signal reset mode
        self.num_steps = 0
        self.molMDP.reset()
        self.reward.reset()

        action_mask = self._get_action_mask()
        self._prev_obs = dict({"action_mask": action_mask})

        if episode is None:
            rnd_state = self._env_seed()
        else:
            rnd_state = self._env_seed(episode)

        for i in range(self.random_steps):
            actions = self._get_av_actions(action_mask)
            action = rnd_state.choice(actions)
            self._reset_step(action)
            action_mask = self._get_action_mask()
            self._prev_obs = dict({"action_mask": action_mask})

            if self._if_terminate():
                print("bad molecule init: resetting MDP")
                self.molMDP.reset()

        self.num_steps = 0
        self._reset_mode = False

        obs, graph = self._make_obs()
        return obs

    def _reset_step(self, action):
        if (action == 0):
            agent_stop = True
        elif action <= (self.max_blocks - 1):
            self.molMDP.remove_jbond(jbond_idx=action-1)
        else:
            stem_idx = (action - self.max_blocks) // self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.molMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

    def step(self, action: int):
        act_smiles = self.molMDP.molecule.smiles

        if not self._prev_obs["action_mask"][action]:
            warnings.warn(f'illegal action: {action} - available {np.sum(self._prev_obs["action_mask"])}')
            return self._prev_obs, 0, True, dict({
                "error": "bad_action", "episode": self._crt_episode, "act_molecule": act_smiles,
                "num_steps": self.num_steps
            })

        if action == 0:
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
        obs, graph = self._make_obs()
        env_stop = self._if_terminate()

        molecule = self.molMDP.molecule

        molecule.graph = graph
        reward, log_vals = self.reward(molecule, agent_stop, env_stop, self.num_steps)

        smiles = self.molMDP.molecule.smiles

        info = {"act_molecule": act_smiles, "res_molecule": smiles, "log_vals": log_vals,
                "num_steps": self.num_steps,
                "mol": self.molMDP.dump()}
        if self._crt_episode is not None:
            info["episode"] = self._crt_episode

        done = any((agent_stop, env_stop))

        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.molMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = self.num_steps
        return mol_attr, num_steps

    def set_state(self, state):
        mol_attr, self.num_steps = state
        [setattr(self.molMDP.molecule, key, deepcopy(value)) for key, value in mol_attr.items()]
        self.molMDP.molecule.blocks = [self.molMDP.block_mols[idx] for idx in state[0]["blockidxs"]]
        self.molMDP.molecule._mol = None
        obs, graph = self._make_obs()
        return obs

    def render(self, outpath):
        # todo: also visualize building blocks used ?
        mol = self.molMDP.molecule.mol
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)

    def _get_av_actions(self, action_mask):
        return self._action_list[action_mask]

    def _get_action_mask(self):
        mol = self.molMDP.molecule
        action_mask = np.zeros(self.max_blocks + self.max_branches * self.num_blocks, dtype=np.bool)
        action_mask[0] = 1  # Action terminate
        if self.allow_removal:
            action_mask[:len(mol.jbonds) + 1] = 1

        if self.molMDP.molecule.numblocks == 0:
            action_mask[self.max_blocks:] = 1  # allow to add any block
        else:
            action_mask[self.max_blocks: (self.max_blocks + len(mol.stems) * self.num_blocks)] = 1

        return action_mask

    def _reset_load(self, state):
        self._crt_episode = None

        self.num_steps = 0
        self.molMDP.reset()
        self.reward.reset()

        self.molMDP.load(state)

        action_mask = self._get_action_mask()
        self._prev_obs = dict({"action_mask": action_mask})
        obs, graph = self._make_obs()
        self.num_steps = 0

        return obs
