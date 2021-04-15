import numpy as np
from scipy import special
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()

def on_episode_start(info):
    # save env state when an episode starts
    env = info["env"].get_unwrapped()[0]
    state = env.get_state()
    episode = info["episode"]
    episode.user_data["initial_state"] = state

class BoltzmannPolicy(Policy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        Policy.__init__(self, observation_space, action_space, config)
        # keep a separate copy of the environment
        self.env = _global_registry.get(ENV_CREATOR, config["env"])(config["env_config"])
        self.temperature = config["env_config"]["temperature"]

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        actions = []
        for i, episode in enumerate(episodes):
            if episode.length == 0:
                # if first time step of episode, get initial env state
                env_state = episode.user_data["initial_state"]
            else:
                # state = self.env.get_state()
                env_state = episode.user_data["current_state"]
            self.obs = self.env.set_state(env_state)
            acts = np.where(self.obs["action_mask"])[0]
            values = []

            for a in enumerate(acts):
                temp_obs, reward, _, info = self.env.step(a)
                # print(reward)
                values.append(reward)
                self.env.set_state(env_state)
            probs = special.softmax(np.divide(values, self.temperature))
            a = acts[np.random.choice(acts.shape[0], p=probs)]
            self.obs, reward, _, info = self.env.step(a)
            episode.user_data["current_state"] = self.env.get_state()
            actions.append(a)

        return np.array(actions), [], {}


    def learn_on_batch(self, samples):
        return {}  # return stats

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass

BoltzmannTrainer = build_trainer(
    name="Boltzmann_Trainer",
    default_policy=BoltzmannPolicy)