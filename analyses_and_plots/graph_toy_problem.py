"""
A simple toy problem unrelated to LambdaZero to show how pytorch_geometric
objects can be passed to the underlying actor-critic model using gym.Dict space.
"""
import itertools
import time
import zlib
from collections import OrderedDict
from pathlib import Path

import torch.nn.functional as F
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
from torch_geometric.nn import NNConv, Set2Set

from copy import copy

import pandas as pd
import matplotlib.pyplot as plt

from LambdaZero.utils import get_external_dirs

MAX_NUMBER_OF_NODES = 40
MAX_NUMBER_OF_EDGES = MAX_NUMBER_OF_NODES * (
    MAX_NUMBER_OF_NODES - 1
)  # 2 x (number_of_nodes choose 2)

NUMBER_OF_NODE_FEATURES = 8
NUMBER_OF_EDGE_FEATURES = 4

BUFFER_SIZE = 8096 * 2


def fast_from_data_list(
    data_list, inckeys=set(["stem_atmidx", "edge_index", "jbond_atmidx"])
):
    """
    I have to copy this method because importing it from its original location breaks with error:
    from ray.rllib.agents.dqn.distributional_q_model import DistributionalQModel
    ModuleNotFoundError: No module named 'ray.rllib.agents.dqn.distributional_q_model'
    """

    keys = data_list[0].keys

    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}

    for key in keys:
        batch[key] = []

    cumsum = [0] * len(keys)
    batch.batch = []
    for j, key in enumerate(keys):
        cat_dim = 1 if key == "edge_index" else 0
        slc = batch.__slices__[key]
        bk = batch[key]
        for i, data in enumerate(data_list):
            item = data[key]
            if cumsum[j] > 0:
                item = item + cumsum[j]
            slc.append(item.shape[cat_dim] + slc[-1])
            if key in inckeys:
                cumsum[j] += data.x.shape[0]
            bk.append(item)
            if j == 0:
                batch.batch.append(torch.full((data.x.shape[0],), i, dtype=torch.long))

    for key in batch.keys:
        item = batch[key][0]
        cd = data_list[0].__cat_dim__(key, item)
        batch[key] = torch.cat(batch[key], dim=cd)

    return batch.contiguous()


def pack(g):
    """
    Slightly modified copy of the pack method of MolGraphSpace class, unthethered to a class. Hardcoding some values
    to what they should be in this toy example.
    """
    attributes = ["x", "edge_index", "edge_attr"]
    _ndims = [2, 2, 2]

    msg = b""
    shapes = np.int16(np.hstack([getattr(g, i).shape for i in attributes])).tostring()
    msg += shapes
    msg += b"".join([getattr(g, i).numpy().tobytes() for i in attributes])
    msg = zlib.compress(msg)
    buf = np.zeros(BUFFER_SIZE, dtype=np.uint8)
    buf[:2] = np.frombuffer(np.uint16(len(msg)).data, np.uint8)
    buf[2 : 2 + len(msg)] = np.frombuffer(msg, np.uint8)
    return buf


def unpack(buf):
    """
    Slightly modified copy of the unpack method of MolGraphSpace class, unthethered to a class. Hardcoding some values
    to what they should be in this toy example.
    """
    attributes = ["x", "edge_index", "edge_attr"]
    _ndims = [2, 2, 2]
    _dtypes = [np.float32, np.int64, np.float32]
    _shapeslen = 12

    _dsizes = [i().itemsize for i in _dtypes]

    l = np.fromstring(buf[:2], np.uint16)[0]
    msg = zlib.decompress(buf[2 : 2 + l])
    d = {}
    shapes = np.fromstring(msg[:_shapeslen], np.int16)
    idx = _shapeslen
    for attr, ndim, dtype, dsize in zip(attributes, _ndims, _dtypes, _dsizes):
        shape = shapes[:ndim]
        l = shape[0] * (1 if ndim == 1 else shape[1]) * dsize
        d[attr] = torch.from_numpy(
            np.frombuffer(msg[idx : idx + l], dtype).reshape(shape)
        )
        idx += l
        shapes = shapes[ndim:]
    return Data(**d)


def make_random_graph():
    """
    This  method just creates a random graph with a random number of nodes
    and random edges. Nodes and edges will have random features.
    """

    number_of_nodes = np.random.randint(
        2, MAX_NUMBER_OF_NODES + 1
    )  # between 2 and MAX_NUMBER_OF_NODES

    maximum_number_of_edges = (
        number_of_nodes * (number_of_nodes - 1) // 2
    )  # (number_of_nodes choose 2)

    number_of_edges = np.random.randint(
        1, maximum_number_of_edges + 1
    )  # between 1 and maximum_number_of_edges
    node_indices = list(range(number_of_nodes))
    all_possible_pairs = np.array(list(itertools.combinations(node_indices, 2)))
    edge_pair_indices = np.random.choice(
        range(len(all_possible_pairs)), number_of_edges, replace=False
    )
    edge_pairs = all_possible_pairs[edge_pair_indices]
    edge_index = torch.from_numpy(
        np.concatenate([edge_pairs, edge_pairs[:, ::-1]]).transpose()
    )
    x = torch.rand(number_of_nodes, NUMBER_OF_NODE_FEATURES, requires_grad=False)
    e = torch.rand(
        number_of_edges, NUMBER_OF_EDGE_FEATURES, requires_grad=False
    )  # factor of 2 because of directed edges
    edge_attr = torch.cat([e, e])
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return graph


def make_dict_from_graph(graph):
    """
    Package a graph as a dict, handling the zero padding as necessary.

    Note that ray uses numpy arrays everywhere under the hood, which breaks
    a bunch of things if I try to put torch.tensors here. This is why I'm passing
    numpy arrays.
    """
    zero_padded_x = np.zeros(
        [MAX_NUMBER_OF_NODES, NUMBER_OF_NODE_FEATURES], dtype=np.float32
    )
    zero_padded_edge_attr = np.zeros(
        [MAX_NUMBER_OF_EDGES, NUMBER_OF_EDGE_FEATURES], dtype=np.float32
    )

    zero_padded_edge_index = np.zeros([2, MAX_NUMBER_OF_EDGES], dtype=np.int)

    x = graph.x
    zero_padded_x[: (len(x)), :] = x

    edge_index = graph.edge_index
    zero_padded_edge_index[:, : edge_index.shape[1]] = edge_index
    zero_padded_edge_attr[: edge_index.shape[1], :] = graph.edge_attr

    cutoff = (
        np.int32((len(x),)),
        np.int32((edge_index.shape[1],)),
    )  # this variable indicates where zero-padding starts.

    buffer = pack(graph)
    return OrderedDict(
        {
            "cutoff": cutoff,
            "x": zero_padded_x,
            "edge_index": zero_padded_edge_index,
            "edge_attr": zero_padded_edge_attr,
            "pack_buffer": buffer,
        }
    )


def make_graphs_from_dict(data_dict: OrderedDict):
    """
    Extract a graph from a dict, taking zero-padding into account.
    """
    graphs = []
    x_cutoff, edge_cutoff = data_dict["cutoff"]
    # RLLib doesn't seem to be respecting the observation space's
    # dtype arguments, so we have to cast to the right type by hand.
    # Plus, gym can't handle zero-dimensional boxes, or I'm doing it
    # wrong, either way a squeeze is necessary to undo the extra dimension
    x_cutoff = x_cutoff.long().squeeze(1)
    edge_cutoff = edge_cutoff.long().squeeze(1)
    for i in range(data_dict["x"].shape[0]):
        x = data_dict["x"][i, : x_cutoff[i], :]
        edge_index = data_dict["edge_index"][i, :, : edge_cutoff[i]].long()
        edge_attr = data_dict["edge_attr"][i, : edge_cutoff[i], :]
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    return graphs


class DictGraphSpace(spaces.Dict):
    def sample(self):
        """
        Overload the sample method to create sane graphs, not random Dict that cannot easily be interpreted as graphs.
        """
        graph = make_random_graph()
        dict = make_dict_from_graph(graph)
        return dict


SPACE = OrderedDict(
    {
        "cutoff": spaces.Tuple(
            (
                spaces.Box(low=0, high=MAX_NUMBER_OF_NODES, shape=(1,), dtype=np.int32),
                spaces.Box(low=0, high=MAX_NUMBER_OF_EDGES, shape=(1,), dtype=np.int32),
            )
        ),
        "x": spaces.Box(
            low=0,
            high=1,
            shape=(MAX_NUMBER_OF_NODES, NUMBER_OF_NODE_FEATURES),
            dtype=np.float32,
        ),
        "edge_index": spaces.Box(
            low=0,
            high=MAX_NUMBER_OF_NODES,
            shape=(2, MAX_NUMBER_OF_EDGES),
            dtype=np.int32,
        ),
        "edge_attr": spaces.Box(
            low=0,
            high=1,
            shape=(MAX_NUMBER_OF_EDGES, NUMBER_OF_EDGE_FEATURES),
            dtype=np.float32,
        ),
        "pack_buffer": spaces.Box(
            low=0, high=256, shape=(BUFFER_SIZE,), dtype=np.int32
        ),
    }
)

DICT_SPACE = DictGraphSpace(SPACE)


class DummyGraphEnv(gym.Env):
    def __init__(self, env_config):
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
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
        reward = np.random.random()
        done = self.counter >= 3

        return obs, reward, done, dict(counter=self.counter)


class ToyGraphActorCriticModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # I don't know how to propagate this path as a parameter from the outside
        # to here through model_config. Ray's config parameters are obscure.
        file_path = Path(__file__).parent.joinpath("forward_timing.csv")
        print(f"output file is {file_path}")
        self.file = open(file_path, "w")

        self.file.write(
            "device,number_of_graphs,unpacking graphs time (s),make graphs from dict time (s),batch time (s),model time (s)\n"
        )

        self.model = DummyMPNN(
            num_node_features=NUMBER_OF_NODE_FEATURES,
            num_edge_features=NUMBER_OF_EDGE_FEATURES,
            dim=64,
        )

    def forward(self, input_dict, state, seq_lens):
        data_dict_observation = input_dict["obs"]

        buf = data_dict_observation.pop("pack_buffer")
        t1 = time.time()
        enc_graphs = buf.data.cpu().numpy().astype(np.uint8)
        buffer_graphs = [unpack(i) for i in enc_graphs]
        t2 = time.time()
        unpacking_time = t2 - t1

        t1 = time.time()
        dict_graphs = make_graphs_from_dict(data_dict_observation)
        t2 = time.time()
        dict_graph_time = t2 - t1

        t1 = time.time()
        data = fast_from_data_list(dict_graphs)
        t2 = time.time()
        batch_time = t2 - t1

        t1 = time.time()
        scalar_outs, _ = self.model(data)
        t2 = time.time()
        model_time = t2 - t1
        self.file.write(
            f"{data.x.device}, {data.num_graphs}, {unpacking_time}, {dict_graph_time}, {batch_time}, {model_time}\n"
        )
        self.file.flush()

        # Ray should put the data on the correct device.
        #  the code in /ray/rllib/policy/torch_policy.py:: line 414
        # is executed (I checked by following breakpoints). This code
        # transforms the numpy array to torch tensors and puts them on
        # self.device.

        self._value_out = torch.zeros(len(dict_graphs))
        return scalar_outs, state

    def value_function(self):  # So PPO doesn't crash
        return self._value_out


class DummyMPNN(nn.Module):
    def __init__(self, num_node_features=14, num_edge_features=4, dim=64):
        super().__init__()
        self.lin0 = nn.Linear(num_node_features, dim)
        net = nn.Sequential(
            nn.Linear(num_edge_features, dim), nn.LeakyReLU(), nn.Linear(dim, dim * dim)
        )
        self.conv = NNConv(dim, dim, net, aggr="mean")
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=1)
        # 2 = [v, simulate logit]
        self.lin_out = nn.Linear(dim * 2, 2)

    def forward(self, data):
        out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(6):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = self.lin_out(out)
        return out, data


def plot_timing_results(data_df):
    """
    Function to plot the timing results
    """
    data_df.columns = ["device", "n", "buffer", "dict", "batch", "model"]
    device = data_df["device"].unique()[0]

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    bin_size = 25
    for n, ax in zip([1, 72, 128], [ax1, ax2, ax3]):
        ax.set_xscale("log")

        df = copy(data_df[data_df["n"] == n])

        df["buffer time per graph"] = df["buffer"] / n
        df["dict time per graph"] = df["dict"] / n
        df["batching time per graph"] = df["batch"] / n
        df["model time per graph"] = df["model"] / n

        df.hist(
            "dict time per graph",
            bins=bin_size,
            ax=ax,
            alpha=0.75,
            label="construct graphs from dictionary",
        )
        df.hist(
            "buffer time per graph",
            bins=bin_size,
            ax=ax,
            alpha=0.60,
            label="construct graphs from unpacked buffer",
        )
        df.hist(
            "batching time per graph",
            bins=bin_size,
            ax=ax,
            alpha=0.25,
            label="create batch from list of graphs [fast_from_data_list]",
        )
        df.hist(
            "model time per graph",
            bins=bin_size,
            ax=ax,
            alpha=0.25,
            label="execute NN model",
        )
        ax.set_ylabel("count")
        ax.set_title(f"batch size = {n}")

    ax1.legend(loc=0)
    ax3.set_xlabel("execution time,  normalized per graph in batch (s)")
    ax1.set_xlim(ax3.get_xlim())
    ax2.set_xlim(ax3.get_xlim())
    fig.suptitle(f"timing tests of forward method, device = {device}")
    return fig


if __name__ == "__main__":
    _, _, summaries_dir = get_external_dirs()

    config = dict(
        env=DummyGraphEnv,
        model={"custom_model": "ToyGraphActorCriticModel"},
        train_batch_size=128,  # use the smallest possible batch so things will run fast. Default is 4000, and slooow.
        lr=1e-4,
        use_pytorch=True,
        num_workers=0,
        num_gpus_per_worker=0,
        num_gpus=0,
    )

    ray.init(local_mode=True)
    ModelCatalog.register_custom_model(
        "ToyGraphActorCriticModel", ToyGraphActorCriticModel
    )

    tune.run(
        PPOTrainer,
        config=config,
        local_dir=summaries_dir,
        stop={"training_iteration": 1},
        name="graph_toy_model",
    )

    # plot the timing results
    local_dir = Path(__file__).parent
    file_path = str(local_dir.joinpath("forward_timing.csv"))
    data_df = pd.read_csv(file_path)
    fig = plot_timing_results(data_df)
    fig.savefig(str(local_dir.joinpath("timing_tests.png")))
