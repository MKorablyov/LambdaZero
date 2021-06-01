import os.path as osp
from LambdaZero.environments import BlockMolEnv_v4, BlockMolEnvGraph_v1, BlockMolEnv_v3, reward
from LambdaZero.environments.block_mol_v3 import synth_config
import LambdaZero.utils

from ray import tune
import os.path as osp

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = {
    "boltzmann_config": {
        "env": BlockMolEnv_v3,
        "env_config": {
            "reward_config": {
                "qed_cutoff": [0., 0.5],
                "synth_cutoff": [0, 4],
                "synth_config": synth_config,
                "soft_stop": True,
                "exp": None,
                "delta": False,
                "simulation_cost": 0.0,
                "device": "cuda",
            },
            "reward": reward.SynthQEDReward,
            "random_blocks": 20,
            "max_atoms": 50,
        }, # can change the reward if needed
        "temperature": 0.11,
    },
    "stop": {"training_iteration": 100},
    "reuse_actors": True,
    "num_samples": 10000,

    "summaries_dir": summaries_dir,
    "memory": 50 * 10 ** 9, # 50
    "object_store_memory": 50 * 10 ** 9,

    "resources_per_trial": {
        # "cpu": 5,
        "cpu": 4,
        "gpu": 0.4,
    }, # requests 40 cpus and 4 gpus
}


# small
boltzmann_config_001_ccr5 = { #
    "boltzmann_config": {
        "apollo_pipe": "ccr5",
    },
}

boltzmann_config_001_int = {
    "boltzmann_config": {
        "apollo_pipe": "int",
    },
}

boltzmann_config_001_rt = {
    "boltzmann_config": {
        "apollo_pipe": "rt",
    }
}

# large datset
boltzmann_config_002_ccr5 = { #
    "boltzmann_config": {
        "env_config": {
            "molMDP_config": {
                "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"),
            },
        },
        "apollo_pipe": "ccr5",
    }
}

boltzmann_config_002_int = { #
    "boltzmann_config": {
        "env_config": {
            "molMDP_config": {
                "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"),
            },
        },
        "apollo_pipe": "int",
    }
}

boltzmann_config_002_rt = { #
    "boltzmann_config": {
        "env_config": {
            "molMDP_config": {
                "blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json"),
            },
        },
        "apollo_pipe": "rt",
    }
}