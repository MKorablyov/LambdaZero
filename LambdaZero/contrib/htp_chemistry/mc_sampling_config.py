import os.path as osp
from ray import tune
from LambdaZero.contrib.htp_chemistry.mc_reactions import rdkit_bicomponent_reactions_v0, rdkit_monocomponent_reactions_v0
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

mc_sampling_config_000 = {
    "mols": osp.join(datasets_dir, "htp_buildingblocks/Enamine_sp3_rich.csv"),
    "bicomponent_reactions": rdkit_bicomponent_reactions_v0,
    "monocomponent_reactions": rdkit_monocomponent_reactions_v0,
    "num_reaction_steps": tune.grid_search([1, 2, 3]) , # 2
    "update_frequency": 500,
    "convergence_check_frequency": 50000/500,  # 1000 for 1 step, 10000 for 2 steps, 100000 for 3 steps
    "convergence_criteria": 0.00001,  # for 1 step: 0.000001, for 2 steps: 0.000001, for 3 steps: 0.000000001 # for 1a 0.00001
    "qed_cutoff": tune.grid_search([0., 0.3]), #0.3
    "multi_substitution": False, # whether to consider all substitutions
    "generate_molecules_mode": None,  # if not None, then update_freq/convcriteria is obsolete; stops whenever this number of molecules is generated
    "wandb": {
        "api_key_file": osp.join(summaries_dir,"wandb_key"),
        "project": "htp_chemistry",
    },
}

mc_sampling_config_001 = {
    "mols": osp.join(datasets_dir, "htp_buildingblocks/mcule_fragment_hsp.csv"),
    "bicomponent_reactions": rdkit_bicomponent_reactions_v0,
    "monocomponent_reactions": rdkit_monocomponent_reactions_v0,
    "num_reaction_steps": tune.grid_search([1, 2, 3]) , # 2
    "update_frequency": 500,
    "convergence_check_frequency": 50000/500,  # 1000 for 1 step, 10000 for 2 steps, 100000 for 3 steps
    "convergence_criteria": 0.00001,  # for 1 step: 0.000001, for 2 steps: 0.000001, for 3 steps: 0.000000001 # for 1a 0.00001
    "qed_cutoff": tune.grid_search([0., 0.3]),
    "multi_substitution": False, # whether to consider all substitutions
    "generate_molecules_mode": None,  # if not None, then update_freq/convcriteria is obsolete; stops whenever this number of molecules is generated
    "wandb": {
        "api_key_file": osp.join(summaries_dir,"wandb_key"),
        "project": "htp_chemistry",
    },
}

mc_sampling_config_002 = {
    "mols": osp.join(datasets_dir, "htp_buildingblocks/mcule_fragment_hsp.csv"),
    "bicomponent_reactions": rdkit_bicomponent_reactions_v0,
    "monocomponent_reactions": rdkit_monocomponent_reactions_v0,
    "num_reaction_steps": 2, # 2
    "update_frequency": 500,
    "convergence_check_frequency": 50000/500,  # 1000 for 1 step, 10000 for 2 steps, 100000 for 3 steps
    "convergence_criteria": 0.00001,  # for 1 step: 0.000001, for 2 steps: 0.000001, for 3 steps: 0.000000001 # for 1a 0.00001
    "qed_cutoff": 0.3,
    "multi_substitution": False, # whether to consider all substitutions
    "generate_molecules_mode": None,  # if not None, then update_freq/convcriteria is obsolete; stops whenever this number of molecules is generated
    "wandb": {
        "api_key_file": osp.join(summaries_dir,"wandb_key"),
        "project": "htp_chemistry",
    },
}