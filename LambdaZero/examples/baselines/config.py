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
    "reuse_actors": True,
    "num_samples": 200,
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
    "reuse_actors": True,
    "num_samples": 200,
}

