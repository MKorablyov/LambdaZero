
"""
A simple toy problem unrelated to LambdaZero to show how pytorch_geometric
objects can be passed to the underlying actor-critic model using gym.Dict space.
"""
import itertools
from collections import OrderedDict

import gym.spaces.dict
import numpy as np
import ray
import torch
from gym import spaces
from gym.spaces import Box
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from torch_geometric.data import Data, Batch

from LambdaZero.utils import get_external_dirs

MAX_NUMBER_OF_NODES = 10
MAX_NUMBER_OF_EDGES = MAX_NUMBER_OF_NODES*(MAX_NUMBER_OF_NODES-1) # 2 x (number_of_nodes choose 2)


def make_random_graph():
    """
    This  method just creates a random graph with a random number of nodes
    and random edges. The node features will just be x = [1] for each node.
    """

    number_of_nodes = np.random.randint(2, MAX_NUMBER_OF_NODES+1)  # between 2 and MAX_NUMBER_OF_NODES

    maximum_number_of_edges = number_of_nodes * (number_of_nodes - 1) // 2  # (number_of_nodes choose 2)

    number_of_edges = np.random.randint(1, maximum_number_of_edges + 1)  # between 1 and maximum_number_of_edges
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


def make_dict_from_graph(graph):
    """
    Package a graph as a dict, handling the zero padding as necessary.

    Note that ray uses numpy arrays everywhere under the hood, which breaks
    a bunch of things if I try to put torch.tensors here. This is why I'm passing
    numpy arrays.
    """
    zero_padded_x = np.zeros([MAX_NUMBER_OF_NODES, 1], dtype=np.int)
    zero_padded_edge_index = np.zeros([2, MAX_NUMBER_OF_EDGES], dtype=np.int)

    x = graph.x
    zero_padded_x[:(len(x)), :] = x

    edge_index = graph.edge_index
    zero_padded_edge_index[:, :edge_index.shape[1]] = edge_index

    cutoff = np.int32((len(x),)), np.int32((edge_index.shape[1],))  # this variable indicates where zero-padding starts.
    return OrderedDict({'cutoff': cutoff, 'x': zero_padded_x, 'edge_index': zero_padded_edge_index})


def make_graphs_from_dict(data_dict: OrderedDict):
    """
    Extract a graph from a dict, taking zero-padding into account.
    """
    graphs = []
    x_cutoff, edge_cutoff = data_dict['cutoff']
    # RLLib doesn't seem to be respecting the observation space's
    # dtype arguments, so we have to cast to the right type by hand.
    # Plus, gym can't handle zero-dimensional boxes, or I'm doing it
    # wrong, either way a squeeze is necessary to undo the extra dimension
    x_cutoff = x_cutoff.long().squeeze(1)
    edge_cutoff = edge_cutoff.long().squeeze(1)
    for i in range(data_dict['x'].shape[0]):
        x = data_dict['x'][i, :x_cutoff[i], :]
        edge_index = data_dict["edge_index"][i, :, :edge_cutoff[i]]
        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs


class DictGraphSpace(spaces.Dict):
    def sample(self):
        """
        Overload the sample method to create sane graphs, not random Dict that cannot easily be interpreted as graphs.
        """
        graph = make_random_graph()
        dict = make_dict_from_graph(graph)
        return dict


SPACE = OrderedDict({
    "cutoff": spaces.Tuple((spaces.Box(low=0, high=MAX_NUMBER_OF_NODES, shape=(1,), dtype=np.int32),
                            spaces.Box(low=0, high=MAX_NUMBER_OF_EDGES, shape=(1,), dtype=np.int32),)),
    "x": spaces.Box(low=0, high=1, shape=(MAX_NUMBER_OF_NODES, 1), dtype=np.int),
    "edge_index": spaces.Box(low=0, high=MAX_NUMBER_OF_NODES, shape=(2, MAX_NUMBER_OF_EDGES), dtype=np.int),
})

#DICT_SPACE = spaces.Dict(SPACE)   # this does bad (not graphs!) sampling??
DICT_SPACE = DictGraphSpace(SPACE) # this breaks ray.preprocessors somehow???


class DummyGraphEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # I'm fiddling with this; it does not currently work.
        self.observation_space = DICT_SPACE
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
        data_dict_observation = input_dict["obs"]

        graphs = make_graphs_from_dict(data_dict_observation)
        data = Batch.from_data_list(graphs)  # Normally we'd pass this to the GraphNN

        logits = torch.zeros(len(graphs), 2)
        self._value_out = torch.zeros(len(graphs))
        return logits, state

    def value_function(self): # So PPO doesn't crash
        return self._value_out


if __name__ == "__main__":
    _, _, summaries_dir = get_external_dirs()
    config = dict(env=DummyGraphEnv,
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

    tune.run(
        PPOTrainer,
        config=config,
        local_dir=summaries_dir,
        stop={"training_iteration": 3},
        name="graph_toy_model",
    )
