import numpy as np
import os.path as osp
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

import LambdaZero.chem
from .molMDP import MolMDP
from rdkit import Chem

import LambdaZero.utils
from .reward import PredDockReward, PredDockReward_v2, PredDockReward_v3

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

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

        # add_block             25 x 105
        # remove_block          25
        # terminate             1

        # molecule [1,2,4,5,8]
        # removal:      MLP(graph_feature[1]) -> 1, 1
        # addition:     MLP(graph_feature[1]) -> 1, 105

        # flatten all actions

        obs = {
            "mol_fp": mol_fp,
            "stem_fps": stem_fps,
            "jbond_fps": jbond_fps,
            "action_mask": action_mask,
            "num_steps": num_steps
               }
        return obs


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
    "reward_config":{
        #"binding_model": osp.join(datasets_dir,"brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth"),
        "binding_model": osp.join(datasets_dir, "brutal_dock/seh/trained_weights/vanilla_mpnn/model.pth"),
        "qed_cutoff": [0.2, 0.7],
        "synth_cutoff": [0, 4],
        "synth_config": synth_config,

        # todo: simulation button; allow dense reward

        "soft_stop": True,
        "exp": None,
        "delta": False,
        "simulation_cost": 0.0,
        "device":"cuda",
    },
    #"reward_config" : {
    #    "soft_stop": True,
    #    "load_model": osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
    #    "natm_cutoff": [45, 50],
    #    "qed_cutoff": [0.2, 0.7],
    #    "exp": None,
    #    "delta": False,
    #    "simulation_cost": 0.00,
    #    "device": "cuda",
    #},
    "reward": PredDockReward_v2,
    "num_blocks": 105, # number of types of building blocks (can only change this to smaller number)
    "max_steps": 7,    # number of steps an agent can take before forced to exit the environment
    "max_blocks": 7,   # maximum number of building blocks in the molecule
    # (if num_bloks(molecule) > max_bloks, block additon is not alowed)
    "max_atoms": 50,
    "max_branches": 25,
    "random_blocks": 2, # regularizer; start environment with molecule with random_bloks bloks
    "max_simulations": 1, # max simulations before agent is forced to exit
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
        reward, log_vals = self.reward(self.molMDP.molecule, simulate, done, self.num_steps)
        info = {"molecule" : self.molMDP.molecule, "log_vals": log_vals}
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
