import time
import numpy as np
import torch
from typing import Dict, List, Optional
from ray.rllib.policy.policy import Policy
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.rllib.utils.typing import AgentID, ModelGradients, ModelWeights, \
    TensorType, Tuple, Union
from ray.rllib.agents.trainer_template import build_trainer
from rdkit import Chem
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor

class RandomPolicy(Policy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        Policy.__init__(self, observation_space, action_space, config)
        self.preprocessor = get_preprocessor(observation_space.original_space)(observation_space.original_space)

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

        # observation is a stacked numpy array; convert into original space
        obs_batch = torch.tensor(obs_batch).float()
        obs_batch = restore_original_dimensions(obs_batch, self.observation_space, "torch")
        action_masks = obs_batch["action_mask"]

        # choose random actions
        actions = []
        for action_mask in action_masks:
            action = np.random.choice(np.where(action_mask)[0])
            actions.append(action)

        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        return {}  # return stats

    def get_weights(self):
        return None
    def set_weights(self, weights):
        assert weights==None, "set weights not supported for Boltzmann Search"



RandomPolicyTrainer = build_trainer(
    name="RandomPolicyTrainer",
    default_policy=RandomPolicy)


