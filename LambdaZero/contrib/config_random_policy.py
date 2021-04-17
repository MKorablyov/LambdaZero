import os.path
import random, string, time
import os.path as osp
from ray.rllib.utils import merge_dicts
import pandas as pd
import LambdaZero.utils
from LambdaZero.contrib.trainer import RandomPolicyTrainer
from LambdaZero.contrib.loggers import log_episode_info
from LambdaZero.contrib.proxy import ProxyUCB
from LambdaZero.contrib.reward import ProxyReward, ProxyRewardSparse
from LambdaZero.contrib.oracle import DockingOracle
from LambdaZero.contrib.inputs import temp_load_data_v1
from LambdaZero.environments import BlockMolEnvGraph_v1
from LambdaZero.contrib.config_model import load_seen_config
from LambdaZero.contrib.config_acquirer import oracle_config, acquirer_config
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

# change default settings for oracle and acquirer
oracle_config = merge_dicts(oracle_config, {"num_threads": 24})
acquirer_config = merge_dicts(acquirer_config, {"acq_size": 256, "kappa": 2.0})


class SaveDocked:
    def __init__(self, outpath=osp.join(summaries_dir, "save_docked")):
        self.outpath = outpath
        if not osp.exists(outpath):os.makedirs(outpath)

    def __call__(self, proposed_x, proposed_d, proposed_acq, x, d, acq, info, y, oname=None):
        oname = oname or ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        smiles, qed, synth_score = [xi["smiles"] for xi in x], [xi["qed"] for xi in x], [xi["synth_score"] for xi in x]
        df = pd.DataFrame(
            {"smiles": smiles, "qed": qed, "synth_score": synth_score, "norm_dockscore": y, "discount": d})
        df.to_feather(os.path.join(self.outpath, oname + ".feather"))
        #print(pd.read_feather(os.path.join(self.outpath, oname + ".feather")))
        return None


proxy_config = {
    "update_freq": 25600,
    "acquirer_config":acquirer_config,
    "oracle": DockingOracle,
    "oracle_config":oracle_config,
    "load_seen": temp_load_data_v1,
    "load_seen_config": load_seen_config,
    "after_acquire":[SaveDocked()],
}


trainer_config = { # tune trainable config to be more precise
    "env": BlockMolEnvGraph_v1,
    "env_config": {
        "molMDP_config": {"blocks_file": osp.join(datasets_dir, "fragdb/pdb_blocks_55.json")},
        "random_steps": 4,
        "allow_removal": True,
        "reward": ProxyReward,
        "reward_config": {
            "synth_options":{"num_gpus":0.05},
            "qed_cutoff": [0.2, 0.5],
            "synth_cutoff":[0, 4],
            "scoreProxy":ProxyUCB,
            "scoreProxy_config":proxy_config,
            "scoreProxy_options":{"num_cpus":1, "num_gpus":1.0},
            "actor_sync_freq": 1500,
        },

    },
    "num_workers": 8,
    "num_gpus_per_worker": 0.15,
    "num_gpus": 0.15,
    "callbacks": {"on_episode_end": log_episode_info},
    "framework": "torch",
    "lr": 5e-5,
    "logger_config":{
        "wandb": {
            "project": "rlbo4",
            "api_key_file": osp.join(summaries_dir, "wandb_key")
        }}}



DEFAULT_CONFIG = {
    "tune_config":{
        "config":trainer_config,
        "local_dir": summaries_dir,
        "run_or_experiment": RandomPolicyTrainer,
        "checkpoint_freq": 25000000,
        "stop":{"training_iteration": 20000},
    },
    "memory": 30 * 10 ** 9,
    "object_store_memory":30 * 10 ** 9
}


debug_config = {
    "memory": 7 * 10 ** 9,
    "object_store_memory": 7 * 10 ** 9,
    "tune_config":{
        "config":{
            "num_workers": 2,
            "num_gpus":0.15,
            "num_gpus_per_worker":0.15,
            "train_batch_size": 256,
            "env_config":{
                "reward_config":{
                    "actor_sync_freq":100,
                    "scoreProxy_options":{"num_cpus":1, "num_gpus":0.3},
                    "scoreProxy_config":{
                        "update_freq": 300,
                        "oracle_config":{"num_threads": 1,},
                        "acquirer_config":{
                            "acq_size": 2,
                            "model_config":{
                                "train_epochs":2,
                                "batch_size":5,
                        }}}}}}}}





rand_001 = {}
