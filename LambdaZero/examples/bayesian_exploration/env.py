from copy import deepcopy

from collections import OrderedDict
from ray.rllib.utils.annotations import override
import cv2
import gym
import numpy as np
from gym.spaces import Discrete, Dict, Box
from ray.rllib.models.preprocessors import Preprocessor, get_preprocessor

class CartPole:
    """
    Wrapper for gym CartPole environment where the reward
    is accumulated to the end
    """

    def __init__(self, config=None):
        self.env = gym.make("CartPole-v0")
        self.action_space = Discrete(2)
        self.observation_space = Dict({
            "obs": self.env.observation_space,
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n, ))
        })
        self.running_reward = 0

    def reset(self):
        self.running_reward = 0
        return {"obs": self.env.reset(), "action_mask": np.array([1, 1])}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.running_reward += rew
        score = self.running_reward if done else 0
        return {"obs": obs, "action_mask": np.array([0, 1])}, score, done, info

    def set_state(self, state):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return {"obs": obs, "action_mask": np.array([1, 1])}

    def get_state(self):
        return deepcopy(self.env), self.running_reward

class Atari:
    """
    Wrapper for gym CartPole environment where the reward
    is accumulated to the end
    """

    def __init__(self, config=None):
        self.env_name = config.get("env_name", "BreakoutNoFrameskip-v4")
        self.env = gym.make(self.env_name)
        self.action_space = self.env.action_space
        # self._grayscale = config.get("grayscale", False)
        # self._zero_mean = config.get("zero_mean", True)
        # self._dim = config.get("dim", 84)
        # if self._grayscale:
        #     shape = (self._dim, self._dim, 1)
        # else:
        #     shape = (self._dim, self._dim, 3)
        # import pdb; pdb.set_trace()
        # if self._zero_mean:
        self.observation_space = Dict({
            "obs": self.env.observation_space,
            "action_mask": Box(low=0, high=1, shape=(self.action_space.n, ))
        })
        # else:
        #     self.observation_space = Dict({
        #         "obs": Box(low=0, high=255, shape=shape),
        #         "action_mask": Box(low=0, high=1, shape=(self.action_space.n, ))
        #     })

    def preprocess(self, observation):
        scaled = observation[25:-25, :, :]
        if self._dim < 84:
            scaled = cv2.resize(scaled, (84, 84))
        # OpenAI: Resize by half, then down to 42x42 (essentially mipmapping).
        # If we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        scaled = cv2.resize(scaled, (self._dim, self._dim))
        if self._grayscale:
            scaled = scaled.mean(2)
            scaled = scaled.astype(np.float32)
            # Rescale needed for maintaining 1 channel
            scaled = np.reshape(scaled, [self._dim, self._dim, 1])
        if self._zero_mean:
            scaled = (scaled - 128) / 128
        else:
            scaled *= 1.0 / 255.0
        return scaled 

    def reset(self):
        # import pdb; pdb.set_trace();
        obs = self.env.reset()
        return {"obs": obs, "action_mask": np.array([1, 1, 1, 1])}

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return {"obs": obs, "action_mask": np.array([1, 1, 1, 1])}, rew, done, info

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = self.env.unwrapped._get_obs()
        return {"obs": obs, "action_mask": np.array([1, 1, 1, 1])}

    def get_state(self):
        return deepcopy(self.env)

class GenericPixelPreprocessor(Preprocessor):
    """Generic image preprocessor.
    Note: for Atari games, use config {"preprocessor_pref": "deepmind"}
    instead for deepmind-style Atari preprocessing.
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space, options):
        self._grayscale = options.get("grayscale")
        self._zero_mean = options.get("zero_mean")
        self._dim = options.get("dim")
        if self._grayscale:
            shape = (self._dim, self._dim, 1)
        else:
            shape = (self._dim, self._dim, 3)

        return shape

    @override(Preprocessor)
    def transform(self, observation):
        """Downsamples images from (210, 160, 3) by the configured factor."""
        self.check_shape(observation)
        scaled = observation[25:-25, :, :]
        if self._dim < 84:
            scaled = cv2.resize(scaled, (84, 84))
        # OpenAI: Resize by half, then down to 42x42 (essentially mipmapping).
        # If we resize directly we lose pixels that, when mapped to 42x42,
        # aren't close enough to the pixel boundary.
        scaled = cv2.resize(scaled, (self._dim, self._dim))
        if self._grayscale:
            scaled = scaled.mean(2)
            scaled = scaled.astype(np.float32)
            # Rescale needed for maintaining 1 channel
            scaled = np.reshape(scaled, [self._dim, self._dim, 1])
        if self._zero_mean:
            scaled = (scaled - 128) / 128
        else:
            scaled *= 1.0 / 255.0
        return scaled.reshape(-1)

class Preprocessor(Preprocessor):
    """Preprocesses each dict value, then flattens it all into a vector.
    RLlib models will unpack the flattened output before _build_layers_v2().
    """

    @override(Preprocessor)
    def _init_shape(self, obs_space, options):
        assert isinstance(self._obs_space, gym.spaces.Dict)
        size = 0
        self.preprocessors = []
        for space in self._obs_space.spaces.values():
            # import pdb; pdb.set_trace();
            if space.shape == (210, 160, 3):
                preprocessor = GenericPixelPreprocessor(space, self._options)
            else:
                preprocessor = get_preprocessor(space)(space, self._options)
            self.preprocessors.append(preprocessor)
            size += preprocessor.size
        return (size, )

    @override(Preprocessor)
    def transform(self, observation):
        self.check_shape(observation)
        array = np.zeros(self.shape)
        self.write(observation, array, 0)
        return array

    @override(Preprocessor)
    def write(self, observation, array, offset):
        if not isinstance(observation, OrderedDict):
            observation = OrderedDict(sorted(observation.items()))
        assert len(observation) == len(self.preprocessors), \
            (len(observation), len(self.preprocessors))
        for o, p in zip(observation.values(), self.preprocessors):
            p.write(o, array, offset)
            offset += p.size

    @property
    @override(Preprocessor)
    def observation_space(self) -> gym.Space:
        obs_space = gym.spaces.Box(-1., 1., self.shape, dtype=np.float32)
        # Stash the unwrapped space so that we can unwrap dict and tuple spaces
        # automatically in model.py
        obs_space.original_space = self._obs_space
        return obs_space