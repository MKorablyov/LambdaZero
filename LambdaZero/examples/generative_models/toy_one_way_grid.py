import argparse
import copy
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

parser.add_argument("--save_path", default='results/test_owg32_3.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-3, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='is_xent', type=str)
parser.add_argument("--momentum", default=0., type=float)
parser.add_argument("--bootstrap_tau", default=0.05, type=float)
parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
parser.add_argument("--horizon", default=32, type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.05, type=float)
parser.add_argument("--device", default='cpu', type=str)
class OneWayGridEnv:

    def __init__(self, horizon, ndim=1, xrange=[-1, 1], func=None):
        self.horizon = horizon
        self.start = np.float32([xrange[0] for i in range(ndim)])
        self.stepsize = (xrange[1]-xrange[0])/self.horizon
        self.end = xrange[1]
        self.ndim = ndim
        self.func = (lambda x: (np.cos(x * 20) + 1) * norm.pdf(x * 5).prod() + 0.01) if func is None else func

    def reset(self):
        self._state = np.concatenate([self.start, [0]],0)
        self._step = 0
        return self._state

    def step(self, a, s=None):
        _s = s
        if _s is None:
            s = self._state
        s, cd = s[:-1], int(s[-1])
        if a == 0:
            s = s + 0
            s[cd] += self.stepsize
        elif a == 1:
            cd += 1
        done = s.max() >= self.end or cd >= self.ndim
        new_s = np.concatenate([s, [cd]],0)
        if _s is None:
            self._state = new_s
            return new_s, 0 if not done else self.func(s).sum(-1), done, {}
        return new_s, 0 if not done else self.func(s).sum(-1), done, {}

    def qvalue_func(self):
        def f(x,):
            r0 = self.step(0, s=x)
            r1 = self.step(1, s=x)
            f0, f1 = (
                f(r0[0]) if not r0[2] else [(r0[0], [r0[1], 0])],
                f(r1[0]) if not r1[2] else [(r1[0], [r1[1], 0])])
            return [(x, [sum(f0[0][-1]), sum(f1[0][-1])])] + f0 + f1
        all_trajectories = sorted(f(self.reset()),
                                  key=lambda x:self._smap[tuple(map(int, x[0]*self.horizon))])
        return all_trajectories

    def all_possible_states(self):
        all_states = [self.reset()]
        """returns a list of all x, and a function to compute p(x) given pi"""
        def f(x, t, xt):
            r0 = self.step(0, s=x)
            r1 = self.step(1, s=x)
            all_states.__iadd__([r0[0], r1[0]])
            i, j = len(all_states) - 2, len(all_states) - 1
            t0 = r0[0], t+[0]
            t1 = r1[0], t+[1]
            return (
                (f(*t0, xt+[i]) if not r0[2] else [(*t0, xt, r0[1])]) +
                (f(*t1, xt+[j]) if not r1[2] else [(*t1, xt, r1[1])]))
        all_trajectories = sorted(f(self.reset(), [], [0]), key=lambda x:tuple(x[0]))
        self._smap = {}
        for i, s in enumerate(all_states):
            self._smap[tuple(map(int, s*self.horizon))] = i
        all_end_states = np.float32([i[0] for i in all_trajectories])
        all_action_trajs = [i[1] for i in all_trajectories]
        all_idx_trajs = [i[2] for i in all_trajectories]
        traj_idxs = [[j]*len(i[2]) for j,i in enumerate(all_trajectories)]
        a = torch.cat(list(map(torch.LongTensor, all_action_trajs)))
        u = torch.cat(list(map(torch.LongTensor, all_idx_trajs)))
        v = torch.cat(list(map(torch.LongTensor, traj_idxs)))
        def compute_all_probs(policy_for_all_states):
            """computes p(x) given pi(a|s) for all s"""
            dev = policy_for_all_states.device
            pi_a_s = torch.log(policy_for_all_states[u, a])
            return torch.exp(
                torch.zeros((len(all_trajectories),), device=dev)
                .index_add(0, v.to(dev), pi_a_s))
        traj_rewards = np.float32([i[3] for i in all_trajectories]).flatten()
        return np.float32(all_states), traj_rewards, all_end_states, compute_all_probs

def make_mlp(l, act=nn.LeakyReLU()):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
        for n, (i, o) in enumerate(zip(l, l[1:]))], []))

def main(args):
    env = OneWayGridEnv(args.horizon)
    dev = torch.device(args.device)
    policy = make_mlp([2] + [args.n_hid] * args.n_layers + [2])
    policy.to(dev)
    q_target = copy.deepcopy(policy)

    opt = torch.optim.Adam(policy.parameters(), args.learning_rate)
    #opt = torch.optim.SGD(policy.parameters(), args.learning_rate, momentum=args.momentum)
    #qq = False


    tf = lambda x: torch.FloatTensor(x).to(dev)
    tl = lambda x: torch.LongTensor(x).to(dev)

    all_states, true_r, all_end_states, compute_p_of_x = env.all_possible_states()
    true_density = tf(true_r / true_r.sum())


    # The chance of having a random episode
    alpha = args.uniform_sample_prob
    num_X = all_states.shape[0]

    losses, visited, distrib_distances = [], [], []
    #inf = torch.tensor(np.inf).to(dev)
    loginf = torch.tensor(1000).to(dev)
    tau = args.bootstrap_tau

    do_td = args.learning_method == 'td'
    do_isxent = args.learning_method == 'is_xent'

    for i in tqdm(range(args.n_train_steps+1)):
        batch = []
        trajs = []
        for j in range(args.mbsize):
            s = env.reset()
            done = False
            is_random = np.random.random() < alpha
            log_probs = []
            while not done:
                pi = Categorical(logits=policy(tf([s])))
                if is_random:
                    a = tl([np.random.randint(2)])
                else:
                    a = pi.sample()
                sp, r, done, _ = env.step(a.item())
                batch.append([tf([i]) for i in (s,a,r,sp,done)])
                log_probs.append(pi.log_prob(a))
                s = sp
            log_p_theta_x = sum(log_probs)
            smoothed_prob = (alpha / num_X  +
                             (1 - alpha) * torch.exp(log_p_theta_x)).detach()
            trajs.append(r / smoothed_prob * log_p_theta_x)

        if do_td:
            s, a, r, sp, d = map(torch.cat, zip(*batch))
            s = s
            sp = sp
            a = a.long()
            q = policy(s)
            with torch.no_grad(): qp = q_target(sp)
            next_q = qp * (1-d).unsqueeze(1) + d.unsqueeze(1) * (-loginf)
            target = torch.logsumexp(torch.cat([torch.log(r).unsqueeze(1), next_q], 1), 1)
            loss = (q[torch.arange(q.shape[0]), a] - target).pow(2).mean()

        elif do_isxent:
            loss = -torch.cat(trajs).mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())

        if do_td and tau > 0:
            for _a,b in zip(policy.parameters(), q_target.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)

        if not i % 50:
            pi_a_s = torch.softmax(policy(tf(all_states)), 1)
            with torch.no_grad():
                estimated_density = compute_p_of_x(pi_a_s)
            k1 = abs(estimated_density - true_density).mean().item()
            print('K1 distance', k1, np.mean(losses[-100:]))
            distrib_distances.append(k1)

    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(losses),
         #'visited': np.float32(visited),
         'policy': policy.to('cpu'),
         'est_d': estimated_density.cpu().numpy(),
         'p_dists': distrib_distances,
         'true_d': true_density.cpu().numpy(),
         'args':args},
        gzip.open(args.save_path, 'wb'))



    if 1:
        return

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
    torch.set_num_threads(1)
    main(args)
