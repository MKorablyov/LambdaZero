from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolEnvGraph_v1
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward
from LambdaZero.contrib.model_with_uncertainty import MolFP
from LambdaZero.contrib.oracle import DockingOracle

model_config = {}

acquirer_config = {
    "model": MolFP,
    "model_config":model_config,
    "acq_size": 32,
    "kappa":0.2
}

oracle_config = {"num_threads":2}

proxy_config = {
    "acquirer_config":acquirer_config,
    "update_freq":100,
    "oracle": DockingOracle,
    "oracle_config":oracle_config
}

rllib_config = {
    "env": BlockMolEnvGraph_v1, # todo: make ray remote environment
    "env_config": {
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyReward,
        "reward_config": {
            "scoreProxy":ProxyUCB,
            "scoreProxy_config":proxy_config,
            "actor_sync_freq":20,
        },

    },
    "num_workers": 8,
    "num_gpus_per_worker": 0.25,
    "num_gpus": 1,
    "model": {
        "custom_model": "GraphMolActorCritic_thv1",
        "custom_model_config": {
            "num_blocks": 105, # todo specify number of blocks only in env?
            "num_hidden": 64
        },
    },
    # "callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics},  # fixme (report all)
    "framework": "torch",
    "lr": 5e-5,
}