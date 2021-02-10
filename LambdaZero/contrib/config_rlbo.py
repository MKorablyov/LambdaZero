import os.path as osp
import torch_geometric.transforms as T
import LambdaZero.utils
import LambdaZero.inputs

from LambdaZero.environments.persistent_search.persistent_buffer import BlockMolEnvGraph_v1
from LambdaZero.environments.reward import PredDockReward_v2
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward,DummyReward
from LambdaZero.contrib.model_with_uncertainty import MolFP
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.inputs import load_data_v1,Mol2GraphProc
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

props = ["dockscore", "smiles"]
transform = T.Compose([LambdaZero.utils.Complete(), LambdaZero.utils.Normalize("dockscore", -8.6, 1.1)])

load_seen_config = {
    "target": "dockscore",
    "dataset_split_path": osp.join(datasets_dir, "brutal_dock/seh/raw/split_Zinc20_docked_neg_randperm_3k.npy"),
    "dataset": LambdaZero.inputs.BrutalDock,
    "dataset_config": {
        "root": osp.join(datasets_dir, "brutal_dock/seh"),
        "props": props,
        "transform": transform,
        "file_names": ["Zinc20_docked_neg_randperm_3k"],
    },
}


model_config = {}

acquirer_config = {
    "model": MolFP,
    "model_config":model_config,
    "acq_size": 32,
    "kappa":0.2
}

oracle_config = {"num_threads":2}

proxy_config = {
    "update_freq":100,
    "acquirer_config":acquirer_config,
    "oracle": DockingOracle,
    "oracle_config":oracle_config,
    "load_seen": load_data_v1,
    "load_seen_config": load_seen_config,
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