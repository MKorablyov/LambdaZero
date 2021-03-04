import os.path as osp

from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyRewardSparse
from LambdaZero.contrib.config_rlbo import proxy_config
from LambdaZero.contrib.htp_chemistry.mc_sampling import MC_sampling_v1a
from LambdaZero.contrib.htp_chemistry.mc_sampling_config import mc_sampling_config_002
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()

htp_env_config_v0_001 = {
    "mc_sampling": MC_sampling_v1a,
    "mc_sampling_config": mc_sampling_config_002,
    "reward": ProxyRewardSparse,
    "reward_config": {
        "synth_cutoff": [-2, -0.5],
        "synth_options": {"num_gpus":0.05},
        "qed_cutoff": [0.2, 0.5],
        # "clip_dockreward": None, # originally 2.5
        "scoreProxy": ProxyUCB,
        "scoreProxy_config": proxy_config,
        "scoreProxy_options": {"num_cpus":2, "num_gpus":1.0},
        "actor_sync_freq": 500,
        # "synth_config": synth_config,
        # "soft_stop": True,
        # "exp": None,
        # "delta": False,
        # "simulation_cost": 0.0,
        "device": "cuda",
    },
    "logger_config": {
        "wandb": {
            "project": "htp_chemistry",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }},
}


htp_env_config_v1_001 = {
    "mc_sampling": MC_sampling_v1a,
    "mc_sampling_config": mc_sampling_config_002,
    "reward": ProxyRewardSparse,
    "reward_config": {
        "synth_cutoff": [-2, -0.5], # no synth_cutoff
        "synth_options": {"num_gpus":0.05},
        "qed_cutoff": [0.2, 0.5],
        # "clip_dockreward": None, # originally 2.5
        "scoreProxy": ProxyUCB,
        "scoreProxy_config": proxy_config,
        "scoreProxy_options": {"num_cpus":2, "num_gpus":1.0},
        "actor_sync_freq": 500,
        # "synth_config": synth_config,
        # "soft_stop": True,
        # "exp": None,
        # "delta": False,
        # "simulation_cost": 0.0,
        "device": "cuda",
    },
    "max_steps": 3,

}