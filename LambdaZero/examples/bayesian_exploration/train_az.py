"""Example of using training on CartPole."""

import argparse

import ray
from ray import tune
from ray.rllib.contrib.alpha_zero.models.custom_torch_models import DenseModel
from env import CartPole
from ray.rllib.models.catalog import ModelCatalog
from LambdaZero.examples.AlphaZero.core.alpha_zero_trainer import AlphaZeroTrainer
from abc import ABC
import numpy as np

from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.agents.callbacks import DefaultCallbacks
torch, nn = try_import_torch()


def convert_to_tensor(arr):
    tensor = torch.from_numpy(np.asarray(arr))
    if tensor.dtype == torch.double:
        tensor = tensor.float()
    return tensor

class AZCallbacks(DefaultCallbacks):
    def on_episode_start(self, worker, base_env, policies, episode):
        # print("episode {} started".format(episode.episode_id))
        episode.user_data["initial_state"] = base_env.get_unwrapped()[0].get_state()

    # def on_episode_end(self, worker, base_env, policies, episode):
    #     env_info = list(episode._agent_to_last_info.values())[0]

    #     for key, value in env_info["log_vals"].items():
    #         episode.custom_metrics[key] = value

class ActorCriticModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space.original_space)(
            obs_space.original_space)

        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        if isinstance(x, torch.Tensor) is False:
            x = x['obs']
        try:
            x = self.shared_layers(x)
        except:
            import pdb; pdb.set_trace();
        # actor outputs
        logits = self.actor_layers(x)

        # compute value
        self._value_out = self.critic_layers(x)
        return logits, None

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with torch.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = torch.squeeze(logits), torch.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value

class DenseModel(ActorCriticModel):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        ActorCriticModel.__init__(self, obs_space, action_space, num_outputs,
                                  model_config, name)

        self.shared_layers = nn.Sequential(
            nn.Linear(
                in_features=obs_space.original_space["obs"].shape[0],
                out_features=256), nn.Linear(
                    in_features=256, out_features=256))
        self.actor_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=action_space.n))
        self.critic_layers = nn.Sequential(
            nn.Linear(in_features=256, out_features=1))
        self._value_out = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=6, type=int)
    parser.add_argument("--training-iteration", default=10000, type=int)
    parser.add_argument("--ray-num-cpus", default=7, type=int)
    args = parser.parse_args()
    ray.init(num_cpus=args.ray_num_cpus)

    ModelCatalog.register_custom_model("dense_model", DenseModel)

    tune.run(
        AlphaZeroTrainer,# "contrib/AlphaZero",
        stop={"training_iteration": args.training_iteration},
        max_failures=0,
        config={
            "env": CartPole,
            "num_workers": args.num_workers,
            "rollout_fragment_length": 50,
            "train_batch_size": 500,
            "sgd_minibatch_size": 64,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 20,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
                "policy_optimization": True
            },
            "ranked_rewards": {
                "enable": True,
            },
            "model": {
                "custom_model": "dense_model",
            },
            "callbacks": AZCallbacks,
        },
    )