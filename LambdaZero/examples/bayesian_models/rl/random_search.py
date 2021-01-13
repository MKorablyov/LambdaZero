import ray
import torch
import numpy as np

from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.trainer_template import build_trainer

class RandomSearchPolicy(Policy):
    """
    Random Search Policy
    """

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        obs = torch.tensor(obs_batch).float()
        obs = restore_original_dimensions(obs, self.observation_space, "torch")
        for i in range(obs['action_mask'].size(0)):
            act_mask = obs['action_mask'][i].numpy()
            act = np.random.choice(len(act_mask))
            while act_mask[act] == 0:
                act = np.random.choice(len(act_mask))
                # act = np.argmax(act_mask == 1) 
            actions.append(act)
        # import pdb;pdb.set_trace();
        return np.array(actions), [], {}

    def learn_on_batch(self, samples):
        return {}  # return stats

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


RandomSearchTrainer = build_trainer(
    name="RandomSearchPolicy",
    default_policy=RandomSearchPolicy)

# ray.init()
# tune.run(MyTrainer, config={"env": "CartPole-v0", "num_workers": 2})