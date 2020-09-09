import numpy as np
import os.path as osp
from copy import deepcopy
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

import LambdaZero.chem
from .chemMDP import chemMDP
from .molMDP import MolMDP
from rdkit import Chem

import LambdaZero.utils
from .reward import PredDockReward, PredDockReward_v2, PredDockReward_v3

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
    "reward_config":{
        "binding_model": osp.join(datasets_dir,"brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth"),
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
    #"max_atoms": 50,
    "max_branches": 25,
    "random_actions": 5, # regularizer; start environment with molecule with random_bloks bloks
    "max_simulations": 1, # max simulations before agent is forced to exit
    "allow_removal": True,

    "max_atoms": 50,

    "add_atom_choices": 7,  # number of atoms to pick from to add
    "edit_atom_choices": 7,  # number of atoms to pick from to edit
    "delete_atom_choices": 1, # just choose which to delete

    "edit_bond_choices": 3, # single, double, triple
    "delete_bond_choices": 1, # just choose which to delete

    "aliphatic_ring_choices": 3+4+5+6+7+8, # number of action choices for aliphatic rings
    "benzene_choices": 6, # number of action choices to add a benzene ring
    # "cylcohexadiene_choices": 6,  #number of action choices to add a cyclohexadiene ring - maybe get rid of this


}

class FPObs_v1:
    def __init__(self, config, ChemMDP):
        self.add_atom_choices = config["add_atom_choices"]
        self.edit_atom_choices = config["edit_atom_choices"]
        self.delete_atom_choices = config["delete_atom_choices"]

        self.edit_bond_choices = config["edit_bond_choices"]
        self.delete_bond_choices = config["delete_bond_choices"]

        self.aliphatic_ring_choices = config["aliphatic_ring_choices"]
        self.benzene_choices = config["benzene_choices"]
        #self.cylcohexadiene_choices = config["cylcohexadiene_choices"]

        # self.atom_actions = config["atom_actions"]
        # self.atom_choices = config["atom_choices"]
        # self.bond_actions = config["bond_actions"]
        # self.bond_choices = config["bond_choices"]
        # self.ring_actions = config["ring_actions"]
        # self.ring_choices = config["ring_choices"]
        # num_actions = self.atom_actions*self.atom_choices + self.bond_actions*self.bond_choices + self.ring_actions*self.ring_choices

        self.max_atoms = config["max_atoms"]
        self.max_steps = config["max_steps"]
        self.ChemMDP = chemMDP


        mol_fp_len = config["obs_config"]["mol_fp_len"] * len(config["obs_config"]["mol_fp_radiis"])
        atom_indx_fp_len = config["obs_config"]["atom_indx_fp_len"] * len(config["obs_config"]["atom_indx_fp_radiis"])


        num_actions = (self.add_atom_choices + self.edit_atom_choices + self.delete_atom_choices ) * self.max_atoms + \
                      (self.edit_bond_choices + self.delete_bond_choices) * (self.max_atoms * 2 - 1)  + \
                      (self.aliphatic_ring_choices + self.benzene_choices ) * self.max_atoms #+ self.cylcohexadiene_choices

        self.action_space = Discrete(num_actions,)
        self.observation_space = Dict({
            "mol_fp": Box(low=0, high=1, shape=(mol_fp_len,)),
            "atom_indx_fps": Box(low=0, high=1, shape=(atom_indx_fp_len,)),
            "num_steps": Discrete(n=self.max_steps + 1),
            "action_mask": Box(low=0, high=1, shape=(num_actions,)),
        })
        self.get_fps = LambdaZero.chem.FPEmbedding_v2(**config["obs_config"])

        #
        # self.num_blocks = config["num_blocks"]
        # self.max_branches = config["max_branches"]
        # self.max_blocks = config["max_blocks"]
        # self.max_atoms = config["max_atoms"]
        # self.max_steps = config["max_steps"]
        # self.molMDP = molMDP
        #
        # num_actions = self.max_blocks + self.max_branches * self.num_blocks
        #
        # self.action_space = Discrete(num_actions,)

    def __call__(self, molecule, num_steps):
        mol_fp, atom_indx_fps_ = self.get_fps(molecule)
        # pad indices
        atom_indx_fps = np.zeros([self.max_branches, atom_indx_fps_.shape[1]], dtype=np.float32)
        atom_indx_fps[:atom_indx_fps_.shape[0], :] = atom_indx_fps_[:self.max_branches, :]
        # make action mask
        break_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        break_mask[:atom_indx_fps_.shape[0]] = 1

        # max number of atoms
        atoms_mask = self.molMDP.block_natm <= (self.max_atoms - molecule.slices[-1])
        branches_mask = self.molMDP.block_nrs <= self.max_branches - len(molecule.stems) - 1
        if len(molecule.jbond_atmidxs) == self.max_blocks - 1:
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
        #
        # mol_fp, stem_fps_, jbond_fps_ = self.get_fps(molecule)
        # # pad indices
        # stem_fps = np.zeros([self.max_branches, stem_fps_.shape[1]], dtype=np.float32)
        # stem_fps[:stem_fps_.shape[0], :] = stem_fps_[:self.max_branches, :]
        # jbond_fps = np.zeros([self.max_blocks - 1, stem_fps_.shape[1]], dtype=np.float32)
        # jbond_fps[:jbond_fps_.shape[0], :] = jbond_fps_[:self.max_blocks-1, :]

        # # make action mask
        # break_mask = np.zeros(self.max_blocks-1, dtype=np.float32)
        # break_mask[:jbond_fps_.shape[0]] = 1

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



class ChemMolEnv_v0:
    mol_attr = ["blockidxs", "slices", "numblocks", "jbonds", "stems", "blockidxs"]

    def __init__(self, config):

        config = merge_dicts(DEFAULT_CONFIG, config)

        self.max_atoms = config["max_atoms"]
        self.add_atom_choices = config["add_atom_choices"]
        self.edit_atom_choices = config["edit_atom_choices"]
        self.delete_atom_choices = config["delete_atom_choices"]

        self.edit_bond_choices = config["edit_bond_choices"]
        self.delete_bond_choices = config["delete_bond_choices"]

        self.aliphatic_ring_choices = config["aliphatic_ring_choices"]
        self.benzene_choices = config["benzene_choices"]
        #self.cylcohexadiene_choices = config["cylcohexadiene_choices"]


        self.max_steps = config["max_steps"]
        self.max_simulations = config["max_simulations"]
        self.random_actions = config["random_actions"]
        #
        self.ChemMDP = chemMDP()
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
        self.ChemMDP.reset()
        self.ChemMDP.random_walk(self.random_actions)
        self.reward.reset()
        return self.observ(self.ChemMDP.molecule, self.num_steps)

    def step(self, action):
        if (action == 0):
            simulate = True
            self.num_simulations += 1
        elif action <= (self.add_atom_choices + self.edit_atom_choices) * self.max_atoms: # add/edit atom
            simulate = False
            atom_indx = action // self.max_atoms
            if self.ChemMDP.molecule
            action_choice = action % 2

        elif action == 2: # edit_atom

        elif action == 3: # delete_atom

        elif action == 4: # edit_bond

        elif action == 5: # delete bond

        elif action == 6: # add ring

        elif action == 7: # aromaticize ring

        elif action <= (self.max_blocks - 1):
            simulate = False
            self.ChemMDP.remove_jbond(jbond_idx=action-1)
        else:
            simulate = False
            stem_idx = (action - self.max_blocks)//self.num_blocks
            block_idx = (action - self.max_blocks) % self.num_blocks
            self.ChemMDP.add_block(block_idx=block_idx, stem_idx=stem_idx)

        self.num_steps += 1
        obs = self.observ(self.ChemMDP.molecule, self.num_steps)
        done = self._if_terminate()
        reward, log_vals = self.reward(self.ChemMDP.molecule, simulate, done, self.num_steps)
        info = {"molecule" : self.ChemMDP.molecule, "log_vals": log_vals}
        return obs, reward, done, info

    def get_state(self):
        mol_attr = {attr: deepcopy(getattr(self.ChemMDP.molecule, attr)) for attr in self.mol_attr}
        num_steps = deepcopy(self.num_steps)
        num_simulations = deepcopy(self.num_simulations)
        previous_reward = deepcopy(self.reward.previous_reward)
        mol = deepcopy(self.ChemMDP.molecule._mol)
        return mol_attr, num_steps, num_simulations, previous_reward, mol

    def set_state(self,state):
        mol_attr, self.num_steps, self.num_simulations, self.reward.previous_reward, self.ChemMDP.molecule._mol \
            = deepcopy(state)
        [setattr(self.ChemMDP.molecule, key, value) for key, value in mol_attr.items()]
        self.ChemMDP.molecule._mol = None
        self.ChemMDP.molecule.blocks = [self.ChemMDP.block_mols[i] for i in self.ChemMDP.molecule.blockidxs]
        return self.observ(self.ChemMDP.molecule, self.num_steps)

    def render(self, outpath):
        mol = self.ChemMDP.molecule
        if mol is not None: Chem.Draw.MolToFile(mol, outpath)


# real docking score
# boltzmann is slow and doesnt do very well somehow, and has errors
# rewrite code
# reward from emmanuel is the dockscore, my is combined, which one should i choose
# right now my action is choosing action+atoms at the same time - a lot of actions actually 3000, how should i choose atoms and then actions
