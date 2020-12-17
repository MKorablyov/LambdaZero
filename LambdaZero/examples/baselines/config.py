from LambdaZero.environments import BlockMolEnv_v4 #, BlockMolEnv_v3
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config
from ray import tune

boltzmann_config_000 = {
    "boltzmann_config": {
        "env": BlockMolEnv_v4,
        "env_config": env_config,
        "temperature": tune.grid_search([0.0001, 0.01, 0.1, 0.15, 0.2, 0.5, 1.0, 2.0, 4.0]),
        "steps": 200,
        "docking": False,
        "env_eval_config":{
            #"dockscore_norm": [-43.042, 7.057]}
            "dockscore_norm": [-8.6, 1.10]},
    },
}

boltzmann_config_001 = {
    "boltzmann_config": {
        "env": BlockMolEnv_v4,
        "env_config": env_config, # can change the reward if needed
        "temperature": 0.15,
        "steps": 200,
        "docking": False,
        "env_eval_config":{
            #"dockscore_norm": [-43.042, 7.057]}
            "dockscore_norm": [-8.6, 1.10]},
    },
}

mcts_config_000 = {
    "mcts_opt_config": {
        "env": BlockMolEnv_v4,
        "env_config": env_config,  # can change the reward if needed
        "steps": 200,
        "docking": False,
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 800,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.020,
            "dirichlet_noise": 0.003,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
            "policy_optimization": False,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
        },
    },
}

