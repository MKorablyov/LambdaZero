import time
import warnings
import os.path as osp
import numpy as np
from rdkit import Chem
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

import LambdaZero.chem
import LambdaZero.utils
from .molMDP import MolMDP
from .reward import PredDockReward, PredDockReward_v2

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

synth_config = {
    "trainer_config": {
        "dataset_type": "regression",
        "train_dataset": None,
        "save_dir": None,

        "features_generator": None,  # Method(s) of generating additional features
        "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
        "no_features_scaling": False,  # Turn off scaling of features
        "num_workers": 8,  # Number of workers for the parallel data loading (0 means sequential)
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
        "num_workers": 8,  # Number of workers for the parallel data loading (0 means sequential)
        "batch_size": 50,  # Batch size
        "disable_progress_bar": True,
        "checkpoint_path": osp.join(datasets_dir, "Synthesizability/MPNN_model/Regression/model_1/model.pt")
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
        # "binding_model": osp.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth"),
        "qed_cutoff": [0.2, 0.7],
        # "synth_cutoff": [0, 4],
        "synth_config": synth_config,
        "soft_stop": True,
        "exp": None,
        "delta": False,
        "simulation_cost": 0.0,
        "device": "cuda",
    },

    # "reward_config": {
    #     "soft_stop": True,
    #     "load_model": osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
    #     "natm_cutoff": [45, 50],
    #     "qed_cutoff": [0.2, 0.7],
    #     "exp": None,
    #     "delta": False,
    #     "simulation_cost": 0.0,
    #     "device": "cuda",
    # },

    "reward": PredDockReward_v2,
    "num_blocks": None, # 105, 464
    "max_steps": 7,
    "max_blocks": 7,
    "max_atoms": 50,
    "max_branches": 20,
    "random_steps": 2,
    "allow_removal": False
}

class BlockMolEnv_v3:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems"]
    def __init__(self, config=None):
        warnings.warn("BlockMolEnv_v3 is deprecated for BlockMolEnv_v4")

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

        self.reward = config["reward"](**config["reward_config"])
        self.get_fps = LambdaZero.chem.FPEmbedding_v2(**config["obs_config"])
        self._prev_obs = None

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

        self._prev_obs = obs

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
        if not action in np.where(self._prev_obs["action_mask"])[0]:
            raise ValueError('Illegal actions')

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
        reward, log_vals = self.reward(self.molMDP.molecule, agent_stop, env_stop, self.num_steps)
        if (self.molMDP.molecule.mol is not None):
            smiles = Chem.MolToSmiles(self.molMDP.molecule.mol)
        else:
            smiles = None
        info = {"molecule": smiles, "log_vals": log_vals}
        #info = {"molecule": self.molMDP.molecule, "log_vals": log_vals}
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




