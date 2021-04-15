import time
import numpy as np
from typing import Dict, List, Optional
from ray.rllib.policy.policy import Policy
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.rllib.utils.typing import AgentID, ModelGradients, ModelWeights, \
    TensorType, TrainerConfigDict, Tuple, Union
from ray.rllib.agents.trainer_template import build_trainer



class BoltzmannPolicy(Policy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        Policy.__init__(self, observation_space, action_space, config)
        # create a local copy of the environment
        self.env = _global_registry.get(ENV_CREATOR, config["env"])(config["env_config"])

        print(config.keys())


        #self.temperature = config["temperature"]

    def compute_actions(
            self,
            obs_batch: Union[List[TensorType], TensorType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorType], TensorType] = None,
            prev_reward_batch: Union[List[TensorType], TensorType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List["MultiAgentEpisode"]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:

        actions = []
        for episode in episodes:
            print(episode)

            actions.append(0)
        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        return {}  # return stats

    def get_weights(self):
        return None
    def set_weights(self, weights):
        assert weights==None, "set weights not supported for Boltzmann Search"



BoltzmannTrainer = build_trainer(
    name="Boltzmann_Trainer",
    default_policy=BoltzmannPolicy)








    # initialize env
    # take some random actions (with dense reward!) (so - will get a whole distribution)
    # I also might need > 40 CPUs
    # after 1st training I want to query about 200 molecules to dock 1
    # max_actions=None
    # temperature
    # env = env(env_config)

    #     def compute_actions(self,
#                         obs_batch,
#                         state_batches=None,
#                         prev_action_batch=None,
#                         prev_reward_batch=None,
#                         info_batch=None,
#                         episodes=None,
#                         **kwargs):
# return actions


#     def learn_on_batch(self, samples):
#         return {}  # return stats
#
#     def get_weights(self):
#         return {}
#
#     def set_weights(self, weights):
#         pass







# import random
# import numpy as np
# import ray
# from LambdaZero.environments import BlockMolEnvGraph_v1
# from LambdaZero.contrib.config_rlbo import trainer_config
#
# ray.init()
# env = BlockMolEnvGraph_v1(trainer_config["env_config"])
#
# for i in range(1000):
#     obs = env.reset()
#     action = np.random.choice(np.where(obs["action_mask"])[0])
#     obs, reward, done, info = env.step(action)
#
#     if done:
#         obs = env.reset()
#     else:
#         done = random.random() > 0.75
#
#     print(info["log_vals"], done)
#     #qed, synth = info["log_vals"], info["log_vals"]["synth"]
#     #print(qed, synth)
#
#
#     #print(qed, synth)
#     #print(step)
#
# #obs, graph = env.step()