"""
A simple toy problem unrelated to LambdaZero to show how pytorch_geometric
objects can be passed to the underlying actor-critic model
"""
import itertools
from collections import OrderedDict

import gym.spaces.dict
import numpy as np
import ray
import torch
import torch_geometric
from gym import Space, ObservationWrapper, spaces
from gym.spaces import Dict, Box
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune import register_env
from torch import nn
from torch_geometric.data import Data

from LambdaZero.utils import get_external_dirs


DICT_SPACE = spaces.Dict({
    "x": spaces.Box(low=0, high=1, shape=(10, 1), dtype=np.int),
    "edge_index": spaces.Box(low=0, high=100, shape=(2, 100), dtype=np.int),
})


class GraphSpace(Space):
    def __init__(self, shape=None, dtype=None):
        super().__init__(shape, dtype)
        self.max_nodes = 10

    def sample(self):
        """This code just creates a random graph with random edges with trivial
        node features .
        """

        number_of_nodes = np.random.randint(2, self.max_nodes)
        maximum_number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2
        number_of_edges = np.random.randint(0, maximum_number_of_edges)
        node_indices = list(range(number_of_nodes))
        all_possible_pairs = np.array(list(itertools.combinations(node_indices, 2)))
        edge_pair_indices = np.random.choice(
            range(len(all_possible_pairs)), number_of_edges, replace=False
        )
        edge_pairs = all_possible_pairs[edge_pair_indices]
        edge_index = torch.from_numpy(
            np.concatenate([edge_pairs, edge_pairs[:, ::-1]]).transpose()
        )
        x = torch.ones(number_of_nodes, 1)
        graph = Data(x=x, edge_index=edge_index)
        return graph

    def contains(self, x):
        """
        Return boolean specifying if x is a valid
        member of this space
        """
        return type(x) == torch_geometric.data.data.Data


class DummyGraphEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # I'm fiddling with this; it does not currently work.
        #self.observation_space = DICT_SPACE #GraphSpace(shape=(2, 1))
        self.observation_space = GraphSpace(shape=(2, 1))
        self.current_state = self.observation_space.sample()
        self.counter = 0

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        """
        Everything here is trivial: the idea is to go through the code with
        break points to see if Ray will accept the data we are trying to feed
        to the Torch model. The implicit RL problem is ill-defined and pointless.
        """
        self.counter += 1

        obs = self.current_state
        reward = 0.0
        done = self.counter >= 3

        return obs, reward, done, dict(counter=self.counter)


class GymWrapper(ObservationWrapper):
    def observation(self, observation):

        # This does not work. A space is not a dict, I don't know how to shove the graph sub-objects
        # into a gym.Dict observation
        #wrapped_observation = Dict({'x': observation.x, 'edge_index': observation.edge_index})
        wrapped_observation = OrderedDict({'x': observation.x, 'edge_index': observation.edge_index})
        return wrapped_observation


def wrapped_env_creator(env_config):
    env = DummyGraphEnv(env_config)
    return GymWrapper(env)


class ToyGraphActorCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        # completely trivial (and useless) model. the model needs to have nn.Module parameters or else
        # ray crashes by saying the optimizer has no parameters to optimize.
        self.fc = nn.Linear(in_features=1, out_features=1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        logits = torch.tensor([0.5, 0.5])
        return logits, None


if __name__ == "__main__":
    _, _, summaries_dir = get_external_dirs()
    config = dict(#env=DummyGraphEnv,#"my_wrapped_env",
                  env="my_wrapped_env",
                  model={"custom_model": "ToyGraphActorCriticModel"},
                  lr=1e-4,
                  use_pytorch=True,
                  num_workers=1,
                  num_gpus_per_worker=0,
                  num_gpus=0)

    ray.init(local_mode=True)
    ModelCatalog.register_custom_model(
        "ToyGraphActorCriticModel", ToyGraphActorCriticModel
    )
    register_env("my_wrapped_env", wrapped_env_creator)

    tune.run(
        PPOTrainer,
        config=config,
        local_dir=summaries_dir,
        stop={"training_iteration": 3},
        name="graph_toy_model",
    )
