import numpy as np
from scipy import special
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, _ = try_import_torch()


class BoltzmannPolicy(TorchPolicy):
    def __init__(self, observation_space, action_space, config, **kwargs):
        super().__init__(
            observation_space, action_space, config
        )
        # keep a separate copy of the environment
        self.env = config["config"]["env"](config['config']["env_config"])
        self.temperature = config['config']["env_config"]["temperature"]
        self.env.reset()
        # self.obs_space = observation_space

    @override(TorchPolicy)
    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        #
        # actions = []
        # obs = torch.tensor(obs_batch).float()
        # obs = restore_original_dimensions(obs, self.observation_space, "torch")
        # for i in range(obs['action_mask'].size(0)):
        #     act_mask = obs['action_mask'][i].numpy()
        #     for act in act_mask:
        #         act = act
        #     act = np.random.choice(len(act_mask))
        #     while act_mask[act] == 0:
        #         act = np.random.choice(len(act_mask))
        #         # act = np.argmax(act_mask == 1)
        #     actions.append(act)

        actions = []

        for i, episode in enumerate(episodes):
            if episode.length == 0:
                # if first time step of episode, get initial env state
                env_state = episode.user_data["initial_state"]
                # create tree root node
                self.obs = self.env.set_state(env_state)

            state = self.env.get_state()
            acts = np.where(self.obs["action_mask"])[0]
            values = []

            for a in enumerate(acts):
                temp_obs, reward, _, info = self.env.step(a)
                print(reward)
                values.append(reward)
                self.env.set_state(state)
            probs = special.softmax(np.divide(values, self.temperature))
            a = acts[np.random.choice(acts.shape[0], p=probs)]
            self.obs, reward, _, info = self.env.step(a)
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