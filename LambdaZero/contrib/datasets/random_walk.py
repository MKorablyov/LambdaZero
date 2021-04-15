""" In order to start Boltzmann Search we need to have some starting point to pretrain proxy.
Here I am sampling some molecules to be docked.
#   filter:
#       num_blocks >= 3
#       QED >= 0.2
#       synth >= 1

"""

# todo: I can make Boltzmann search but fix proxy reward at 1.0
# todo: for Boltzmann we could use Actor + Critic (sample k=100 best actions from actor)
# todo: we can create some special architecture that could be explicitly creating / sampling some futures



import random
import numpy as np
import ray

from LambdaZero.environments import BlockMolEnvGraph_v1

from LambdaZero.contrib.config_rlbo import trainer_config




ray.init()
env = BlockMolEnvGraph_v1(trainer_config["env_config"])

for i in range(1000):
    obs = env.reset()
    action = np.random.choice(np.where(obs["action_mask"])[0])
    obs, reward, done, info = env.step(action)

    if done:
        obs = env.reset()
    else:
        done = random.random() > 0.75

    print(info["log_vals"], done)
    #qed, synth = info["log_vals"], info["log_vals"]["synth"]
    #print(qed, synth)


    #print(qed, synth)
    #print(step)

#obs, graph = env.step()