"""
There seems to be a bottleneck in GraphMolActorCritic_thv1. This script instantiates this class
outside of the ray ecosystem to try to understand what about it is slow.

I use line_profiler to time the code: <https://github.com/pyutils/line_profiler>
with some extra help from this blog: <https://lothiraldan.github.io/2018-02-18-python-line-profiler-without-magic/>

> pip install line_profiler

I also try to document what I understand (or don't!) about the underlying code.
"""
import os
import numpy as np
from collections import namedtuple
import torch.nn.functional as F

import torch
from torch import tensor

from LambdaZero.environments import BlockMolEnvGraph_v1
from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
from LambdaZero.utils import get_external_dirs
from analyses_and_plots.profiling import profile


@profile
def main():

    config = dict(reward_config=dict(device='cpu'))  # debug locally on a non-gpu machine
    mpnn_parametric_config = dict(
        num_hidden=32)  # only one parameter of the model can be changed from this config dict.
    environment = BlockMolEnvGraph_v1(config=config)

    """
    Running a modified version of "train_ppo" in debug mode and putting break points inside the "__init__" method of
    the GraphMolActorCritic_thv1 object, I see that 
        __init__(self, obs_space, action_space, num_outputs, model_config, name)
        
        obs_space:    Box(10211,)       (gym.spaces.box.Box) 
        action_space: Discrete(2107)    (gym.spaces.discrete.Discrete) 
        num_outputs: 2107
        model_config: many parameters, not all provided by me ... (Dict)
        name: "default_model"

    A few notes:
        1) the dimension of the obs_space (10211) is a magic number I can't figure out. Ray shoves the
           observation_space in a Box for whatever reason. I don't think this number is important

        2) The dimension of the action space is given by
           num_actions = self.max_blocks + self.max_branches * self.num_blocks = 2107
                where in the config we have
                    - "num_blocks": 105,
                    - "max_blocks": 7,
                    - "max_branches": 20
            so that explains this magic number.                
    """

    #  Use duck typing to create an object with the correct property. We are "off the rails", ie not
    #  within the ray framework, so this requires a bit of fiddling. The convoluted processing inside of ray
    #  shoves the observation space into a gym.Box somehow. This is not relevant to what we are doing, there is
    #  no sense in trying to reproduce it.
    DuckTypeObservationSpace = namedtuple("DuckTypeObservationSpace", "original_space")

    obs_space = DuckTypeObservationSpace(environment.observation_space)
    action_space = environment.action_space
    number_of_actions = action_space.n

    actor_critic = GraphMolActorCritic_thv1(obs_space=obs_space,
                                            action_space=action_space,
                                            num_outputs=number_of_actions,
                                            model_config=mpnn_parametric_config,
                                            name="time_test_debugging")

    """
    Running a modified version of "train_ppo" in debug mode and putting break points inside the "forward" method of
    the GraphMolActorCritic_thv1 object, I see that 
    
    input_dict =
        {'obs': OrderedDict([('action_mask', tensor([[1., 1., 0., ..., 0., 0., 0.]])),      shape: [1, 2107]
                         ('mol_graph', tensor([[3., 90., 31., ..., 0., 0., 0.]])),          shape: [1, 8096]
                         ('num_steps', tensor([[0., 0., 1., 0., 0., 0., 0., 0.]]))]),       shape: [1, 8]
     'prev_actions': [array(0)],
     'prev_rewards': [0.0],
     'obs_flat': tensor([[1., 1., 0., ..., 0., 0., 0.]])}                                   shape: [1, 10211]
     
     
     state = []                                 (not used internally)
     seq_lens = tensor([1], dtype=torch.int32)  (not used internally)
    
    Notes:
        1) We already saw where 2107 comes from.
        2) 8096 is another magic number. It is hardcoded as the _size property of the MolGraphSpace class. It is 
           not clear where that number comes from. Maybe it's just "large" enough.
        3) It seems the number of steps (8) comes from "max_steps" (7), plus one.
    """

    random_walk_length = 1
    """
     The choice of the random_walk_length is non-trivial: the code crashes with dimension mismatch 
     if random_walk_length > 7 when I try to create a single graph:
        break_logits[i, :len(g.jbond_atmidx)] = g.jbond_preds
    RuntimeError: The expanded size of the tensor (6) must match the existing size (7) at non-singleton dimension 0.  Target sizes: [6].  Tensor sizes: [7]
    
     If I try creating multiple graphs, it breaks quickly with the number of graphs. I guess
     g.jbond_atmidx increases in dimension with the number of samples from the MDP in the environment.
    """

    environment.num_steps = 8  # forcing this variable to be present. It is not used in the model, but it is
                               # needed to generate observations.

    # Create a bunch of random graphs from the environment
    list_numpy_observations = []
    number_of_graphs = 6  # I would like to do more, but the code crashes because of the random_walk issue above
    for _ in range(number_of_graphs):
        environment.molMDP.random_walk(random_walk_length)
        # Create an "observation", and transform it in the format needed for the actor_critic model.
        numpy_obs = environment._make_obs()
        list_numpy_observations.append(numpy_obs)

    # Create the correct object to pass to the model
    obs = {}
    for key in ['mol_graph', 'action_mask']:
        stacked_values = np.array([o[key] for o in list_numpy_observations])
        obs[key] = torch.from_numpy(stacked_values)

    obs['num_steps'] = torch.zeros(number_of_graphs, 8)  # not used so it doesn't really matter

    input_dict = dict(obs=obs)

    state = []
    seq_lens = tensor([1], dtype=torch.int32)

    logits, s = actor_critic.forward(input_dict,
                                     state=state,
                                     seq_lens=seq_lens)

    # Make a dummy loss to do a backwards pass
    zeros = torch.zeros(logits.shape)
    loss = F.mse_loss(logits, zeros)
    loss.backward()


if __name__ == '__main__':
    main()
