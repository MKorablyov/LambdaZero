import numpy as np
import torch
from gym.spaces import Discrete, Dict, Box
from ray.rllib.utils import merge_dicts

def rescale(x, a, b, c, d):
    """ Rescales variable from [a, b] to [c, d]
    """
    return c + ((d - c) / (b - a)) * (x - a)


def schwefel(x, xmin=-1, xmax=1, yrange=2000):
    """
    https://www.sfu.ca/~ssurjano/schwef.html
    """
    x = rescale(x, xmin, xmax, -500, 500)
    val = x.abs().sqrt().sin().mul(x).sum(1).mul(-1).add(418.9829 * x.size(1))
    return val / yrange

DEFAULT_CONFIG = {
    "num_dims": 5,
    "discretize": 15,
    "max_steps": 20,
    "ksize": 0.05,
}

class DummyNonConvex:
    def __init__(self, config):
        config = merge_dicts(DEFAULT_CONFIG, config)

        self._num_dims = config["num_dims"]
        self._discretize = config["discretize"]
        self._ksize = config["ksize"]
        self._max_steps = config["max_steps"]

        # left, stop, right for each dimension to step
        self.action_space = Discrete(2 ** self._num_dims)
        # observation is 1/(1 + distance) for each dimension
        self.observation_space = Box(low=-0.01, high=1.01, shape=(self._num_dims * self._discretize,))

    def _make_obs(self):
        hot_dist = np.tile(np.arange(self._discretize)[None, :],[self._num_dims,1])
        hot_dist = np.abs((hot_dist - self._pos[:, None]) / self._discretize)
        hot_dist = 1. / (1. + hot_dist)
        hot_embedding = np.exp(hot_dist * (1./self._ksize)) / np.exp(1 * (1./self._ksize))
        obs = hot_embedding.reshape(-1)
        return obs

    def _reward(self):
        pos = torch.tensor(self._pos/ self._discretize, dtype=torch.float)[None,:]
        reward = - schwefel(pos).cpu().numpy()[0]
        return reward

    def step(self, action):
        moves = []
        action_ = action
        for div in reversed(2**np.arange(self._num_dims)):
            moves.append(1 if action_//div else -1)
            action_ = action_ - (div*(action_//div))
        self._pos = (self._pos + np.asarray(moves)).clip(0,self._discretize-1)
        obs = self._make_obs()
        reward = self._reward()
        done = True if self.istep == self._max_steps else False
        info = {}
        self.istep +=1
        return obs, reward, done, info

    def reset(self):
        self.istep = 0
        self._pos = np.random.randint(low=0, high=self._discretize, size=self._num_dims)
        return self._make_obs()