import argparse
import gzip
import os
import pickle

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=64, help="Minibatch size", type=int)
parser.add_argument("--tree_horizon", default=5, type=int)
parser.add_argument("--n_hid", default=128, type=int)
parser.add_argument("--n_train_steps", default=1000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.05, type=float)


class BinaryTreeEnv:

    def __init__(self, horizon, xrange=[-1, 1]):
        self.horizon = horizon
        self.center = sum(xrange)/2
        self.width = xrange[1] - xrange[0]
        self.func = lambda x: (np.cos(x * 50) + 1) * norm.pdf(x * 5)

    def reset(self):
        self._state = self.center
        self._step = 0
        return np.float32([self._state])

    def step(self, a):
        self._step += 1
        delta = self.width / (2 ** (self._step + 1))
        if a == 0: # right
            self._state += delta
        elif a == 1:
            self._state -= delta
        done = self._step >= self.horizon or a == 2
        return self._state, 0 if not done else self.func(self._state), done, {}

    def all_possible_states(self):
        """returns a list of all x, and a function to compute p(x) given pi"""
        def f(x, d, t, xt):
            if d >= self.horizon:
                return [(x, t, xt)]
            delta = self.width / (2 ** (d + 2))
            return [(x, t+[2], xt+[x])] + f(x - delta, d + 1, t+[1], xt+[x]) + f(x + delta, d + 1, t+[0], xt+[x])
        all_trajectories = sorted(f(self.center, 0, [], []), key=lambda x:x[0])
        all_states = np.float32([i[0] for i in all_trajectories])
        state_map = {i: j for j, i in enumerate(all_states)}
        all_action_trajs = [i[1] for i in all_trajectories]
        all_idx_trajs = [[state_map[x] for x in xs] for _,_,xs in all_trajectories]
        traj_idxs = [[j]*len(xs) for j,(_,_,xs) in enumerate(all_trajectories)]
        a = torch.cat(list(map(torch.LongTensor, all_action_trajs)))
        u = torch.cat(list(map(torch.LongTensor, all_idx_trajs)))
        v = torch.cat(list(map(torch.LongTensor, traj_idxs)))
        def compute_all_probs(policy_for_all_states):
            """computes p(x) given pi(a|s) for all s"""
            pi_a_s = torch.log(policy_for_all_states[u, a])
            return torch.exp(torch.zeros((len(all_states),)).index_add(0, v, pi_a_s))
        return all_states, compute_all_probs


def make_mlp(l, act=nn.LeakyReLU()):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
        for n, (i, o) in enumerate(zip(l, l[1:]))], []))

def main(args):
    env = BinaryTreeEnv(args.tree_horizon)
    policy = make_mlp([1, args.n_hid, args.n_hid, 3])
    opt = torch.optim.Adam(policy.parameters(), args.learning_rate)

    # The chance of having a random episode
    alpha = args.uniform_sample_prob
    num_X = sum(2 ** i for i in range(args.tree_horizon))

    tf = lambda x: torch.FloatTensor(x)
    tl = lambda x: torch.LongTensor(x)

    all_states, compute_p_of_x = env.all_possible_states()
    true_r = env.func(all_states)
    true_density = true_r / true_r.sum()

    losses, visited = [], []

    for i in tqdm(range(args.n_train_steps)):
        batch = []
        for j in range(args.mbsize):
            s = env.reset()
            done = False
            is_random = np.random.random() < alpha
            log_probs = []
            ac = []
            while not done:
                pi = Categorical(logits=policy(tf([s])))
                if is_random:
                    a = tl([np.random.randint(3)])
                else:
                    a = pi.sample()
                log_probs.append(pi.log_prob(a))
                s, r, done, _ = env.step(a.item())
                ac.append(a)
            log_p_theta_x = sum(log_probs)
            smoothed_prob = alpha / num_X  + (1 - alpha) * torch.exp(log_p_theta_x).detach()
            batch.append((tf([s]), tf([r]), log_p_theta_x, smoothed_prob))
        s, r, log_p_theta_x, smoothed_prob = map(torch.cat, zip(*batch))
        loss = -(r / smoothed_prob * log_p_theta_x).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())
        visited.append(s.numpy())
        if not i % 100:
            pi_a_s = torch.softmax(policy(tf(all_states).unsqueeze(1)), 1)
            with torch.no_grad():
                estimated_density = compute_p_of_x(pi_a_s)
            print('K1 distance', abs(estimated_density - true_density).mean().item())

    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(losses),
         'visited': np.float32(s),
         'policy': policy,
         'est_d': estimated_density,
         'true_d': true_density},
        gzip.open(args.save_path, 'wb'))

    import matplotlib.pyplot as pp
    pp.figure(figsize=(8,4))
    pp.plot(all_states, true_density, label='true p(x)')
    pp.plot(all_states, estimated_density, label='estimated p(x)')
    pp.legend()
    pp.grid()
    pp.xlabel('x')
    pp.ylabel('p(x)')
    pp.title('p(x) propto (cos(x/50) + 1)*norm(x*5)')
    pp.savefig(f'{root}/density.png')

    pp.figure()
    l, = pp.plot(losses, label='train loss', alpha=0.25)
    pp.plot(np.linspace(0,len(losses), 50),
            np.float32(losses).reshape((50, -1)).mean(1),
            c=l.get_c())
    pp.legend()
    pp.xlabel('training steps')
    pp.ylabel('loss')
    pp.grid()
    pp.savefig(f'{root}/loss.png')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
