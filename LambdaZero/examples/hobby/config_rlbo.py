from LambdaZero.environments.persistent_search.persistent_buffer import \
    BlockMolEnvGraph_v1
from LambdaZero.examples.hobby.proxy import ProxyUCB
from LambdaZero.examples.hobby.reward.bayesian_reward_v2 import ProxyReward
from LambdaZero.examples.hobby.inputs import mol_to_graph_v1

proxy_config = {
    "update_freq":20,
    "proc_func":mol_to_graph_v1,
}

rllib_config = {
    "env": BlockMolEnvGraph_v1, # fixme maybe ray.remote the buffer as well
    "env_config": {
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyReward,
        "reward_config": {
            "scoreProxy":ProxyUCB,
            "scoreProxy_config":proxy_config,

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
    #"callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics},  # fixme (report all)
    "framework": "torch",
    "lr": 5e-5,
}