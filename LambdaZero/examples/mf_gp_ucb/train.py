import sys, os, time, socket
import numpy as np

import ray
from ray import tune

import LambdaZero.utils

from LambdaZero.environments.block_mol_v5 import BlockMolEnv_v5
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as DEFAULT_ENV_CONFIG
#from LambdaZero.examples.persistent_search import config
from LambdaZero.models import MPNNet

_, _, summaries_dir = LambdaZero.utils.get_external_dirs()

class TrainableBoltzman(tune.Trainable):
    def _setup(self, config):
        # Environment
        c = DEFAULT_ENV_CONFIG
        c["reward_config"]["device"] = "cpu"
        self.env = BlockMolEnv_v5(c)
        # Reward function
        reward = LambdaZero.environments.reward.PredDockReward(load_model=config['dockscore_model'],
                                natm_cutoff=[45, 50],
                                qed_cutoff=[0.2, 0.7],
                                soft_stop=False,
                                exp=None,
                                delta=False,
                                simulation_cost=0.0,
                                device="cuda")
        self.oracles = [
            lambda mol: reward(mol,env_stop=False,simulate=True,num_steps=1)
        ]
    def step(self):
        if self.done:
            self.obs = self.env.reset()
        actions = np.where(self.obs['action_mask'])[0]
        # TODO: take random action
        action = actions[0]
        self.obs, _, self.done, self.info = self.env.step(actions)
    def _train(self):
        self.step()
        mol = self.info.molecule.mol
        return {'score': self.oracles[0](mol)}
    def _save(self, chkpt_dir):
        return {}

DEFAULT_CONFIG = {
    'trainer': Trainable,
    'config': {
        'dockscore_model': '/home/ml/hhuang63/Summaries/BasicRegressor/BasicRegressor_0_2020-06-02_17-41-06bzogful8/checkpoint_200/model.pth'
    },
    "summaries_dir": summaries_dir,
    "memory": 60 * 10 ** 9,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 100},
}

if __name__=='__main__':
    config = DEFAULT_CONFIG
    config_name = 'DEFAULT'
    analysis = tune.run(
        config['trainer'],
        config=config['config'],
        stop=config["stop"],
        max_failures=0,
        #config=config["rllib_config"],
        local_dir=summaries_dir,
        name=config_name,
        checkpoint_freq=config["checkpoint_freq"],
        num_samples=1
    )

    dfs = analysis.trial_dataframes
