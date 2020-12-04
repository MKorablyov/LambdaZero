import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import torch as th
import seaborn as sns
from gym.spaces import Discrete, Dict, Box
import time
import torch
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
import ray
import LambdaZero.utils
from LambdaZero.environments import DummyNonConvex


_, _, summaries_dir = LambdaZero.utils.get_external_dirs()

DEFAULT_CONFIG = \
    {
        "rllib_config":{
            "tf_session_args": {"intra_op_parallelism_threads": 1, "inter_op_parallelism_threads": 1},
            "local_tf_session_args": {"intra_op_parallelism_threads": 4, "inter_op_parallelism_threads": 4},
            "num_workers": 1,
            "num_gpus_per_worker": 0.1,
            "num_gpus": 0.4,
            #"model": {"custom_model": "MolActorCritic_tfv1"},
            "env": DummyNonConvex,
            "env_config":{"num_dims": 10,
                          "discretize": 7,
                          "max_steps": 10,
                          "ksize": 0.15,
                          }
            #"callbacks": {"on_episode_end": LambdaZero.utils.dock_metrics}
        },
        "memory":  15 * 10**9,
        "trainer": PPOTrainer,
        "checkpoint_freq": 250,
        "stop": {"training_iteration": 2000000},
    }


config = DEFAULT_CONFIG
config_name = "schwefel_ppo"
#config = merge_dicts(DEFAULT_CONFIG, config)


if __name__ == "__main__":
    ray.init(memory=config["memory"])
    tune.run(config["trainer"],
        stop=config["stop"],
        max_failures=0,
        config=config["rllib_config"],
             local_dir=summaries_dir,
             name=config_name,
             checkpoint_freq=config["checkpoint_freq"])


    # config = {}
    # env = DummyNonConvex(config)
    # env.reset()
    #
    # for i in range(100000):
    #     action = np.random.randint(low=0, high=env.action_space.n)
    #     #print(action)
    #     obs, reward, done, info =  env.step(action)
    #     if done: env.reset()
    #     print(reward)


    #from LambdaZero.environments.dummy_nonconvex import schwefel


    # def plot_traj():
    #     x = th.tensor((np.mgrid[-100:100, -100:100] / 100).reshape(2, -1).T)
    #     y = schwefel(x)
    #     x = x.numpy().T.reshape(2, 2 * 100, 2 * 100)
    #     y = y.numpy().reshape(2 * 100, 2 * 100)
    #     plt.subplot(1, 1, 1, projection="3d")
    #     ax = plt.gca()
    #     ax.plot_surface(x[0], x[1], y, cmap='coolwarm', edgecolor='none',
    #                     rstride=1, cstride=1, color='c', alpha=0.5, linewidth=0)
    #
    #     plt.savefig("/home/maksym/Desktop/traj.png", dpi=400)
    #
    #
    # plot_traj()

    #def embedding_distribution(self):
    #    pass
        #sns.distplot(hot_embedding)
        #plt.show()
        #print(hot_embedding)