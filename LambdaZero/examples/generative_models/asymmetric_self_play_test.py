from asymmetric_self_play_agents import *
from toy_1d import BinaryTreeEnv

import argparse
import os
import torch
from torch.distributions import Categorical
import numpy as np
import scipy
from collections import Counter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='testing the ASP framework')

# ====== Loading arguments ======
parser.add_argument('--alice_agent_load_path',
                    default='./asp_models/alice.pt', help='alice agent path info')
parser.add_argument('--bob_agent_load_path',
                    default='./asp_models/bob.pt', help='bob agent load path')

# ====== Environment Arguments ======
parser.add_argument('--horizon', type=int, default=10,
                    help='horizon for tree env')

# ====== Simulation Arguments ======
parser.add_argument('--num_trajs', type=int, default=100,
                    help='number of trajectories to rollout for Alice and Bob')
parser.add_argument('--rollout_agent', default='alice',
                    help='agent to roll out (Alice or Bob)')
parser.add_argument('--seed', type=int, default=100,
                    help='seed for experiments to determine validity')

args = parser.parse_args()

# environment setup
env = BinaryTreeEnv(args.horizon)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# agents
alice_info = torch.load(args.alice_agent_load_path)
bob_info = torch.load(args.bob_agent_load_path)

alice = alice_info['policy']
bob = bob_info['policy']

rollout_agent = alice if args.rollout_agent == 'alice' else bob

final_rewards = []
final_positions = []
for _ in range(args.num_trajs):
    ob = torch.tensor(env.reset()).float()
    done = False

    trajectory = [ob[0].item()]

    while not done:
        with torch.no_grad():
            logits = rollout_agent(ob)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_ob, r, done, _ = env.step(action)
            trajectory.append(next_ob)
            next_ob = torch.tensor([next_ob]).float()

            if done:
                final_rewards.append(r)
                final_positions.append(next_ob[0].item())

            ob = next_ob
    # print(trajectory)

agent_str = 'Alice' if args.rollout_agent == 'alice' else 'Bob'

print(f'After {args.num_trajs} rollouts, the max reward that '
      + agent_str + f' got is {np.max(final_rewards)}.\n')

# mode reflection
reward_modes = scipy.stats.mode(final_rewards)
for mode, num_times in zip(reward_modes[0], reward_modes[1]):
    print(f'After {args.num_trajs} rollouts, one mode reward that '
          + agent_str + f' got is {mode}, which appeared {num_times} times.\n')

# mode reflection position-wise
position_modes = scipy.stats.mode(final_positions)
for mode, num_times in zip(position_modes[0], position_modes[1]):
    print(f'After {args.num_trajs} rollouts, one mode position that '
          + agent_str + f' got is {mode}, which was reached {num_times} times.')

print('FINAL REWARD SUMMARY')
# all modes
unique_rewards = np.unique(final_rewards)
counts = Counter(final_rewards)
for reward in unique_rewards:
    if counts[reward] > 0:
        print(
            f'The {agent_str} GFlowNet received a final reward of {reward} {counts[reward]} times.')

r_counts = [counts[reward] for reward in unique_rewards]
# print(r_counts)

plt.plot(unique_rewards, r_counts)
plt.xlabel('Unique rewards')
plt.ylabel('Reward frequency')
plt.show()

print('FINAL POSITION SUMMARY')
unique_positions = np.unique(final_positions)
counts = Counter(final_positions)
for pos in unique_positions:
    if counts[pos] > 0:
        print(
            f'{agent_str} got to a final position of {pos} {counts[pos]} times.')

p_counts = [counts[pos] for pos in unique_positions]
# print(r_counts)

plt.plot(unique_positions, p_counts)
plt.xlabel('Unique positions')
plt.ylabel('Position frequency')
plt.show()
