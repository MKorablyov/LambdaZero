import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_mars_corners_1.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='is_xent_q', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--bufsize", default=128, help="Buffer size", type=int)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--compute_l1k1", default=0, type=int)
parser.add_argument("--horizon", default=20, type=int)
parser.add_argument("--ndim", default=2, type=int)
parser.add_argument("--n_hid", default=64, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
parser.add_argument('--func', default='corners')

parser.add_argument("--datasource", default='sample', type=str)
parser.add_argument("--n_dataset_pts", default=10000, type=int)
parser.add_argument("--test_ratio", default=0.05, type=float)

# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.00, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

#@profile
def func_corners(x):
    ax = abs(x)
    return (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2 + 1e-1

class GridEnv:

    def __init__(self, horizon, ndim=2, xrange=[-1, 1], func=None):
        self.horizon = horizon
        self.start = [xrange[0]] * ndim
        self.ndim = ndim
        self.width = xrange[1] - xrange[0]
        self.func = (
            (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)).prod(-1) + 0.01)
            if func is None else func)
        self.xspace = np.linspace(*xrange, horizon)

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        return (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)

    def reset(self):
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        return self.obs(), self.func(self.s2x(self._state)), self._state

    def parent_transitions(self, s, used_stop_action):
        if used_stop_action:
            return [self.obs(s)], [self.ndim]
        parents = []
        actions = []
        for i in range(self.ndim):
            if s[i] > 0:
                sp = s + 0
                sp[i] -= 1
                if sp.max() == self.horizon-1: # can't have a terminal parent
                    continue
                parents += [self.obs(sp)]
                actions += [i]
        return parents, actions

    #@profile
    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        sc = s + 0
        if a < self.ndim:
            s[a] = min(s[a]+1, self.horizon-1)
        if a >= self.ndim:
            s[a-self.ndim] = max(s[a-self.ndim]-1,0)

        #sc = np.clip(s, 0, self.horizon-1)
        reverse_a = ((a + self.ndim) % (2 * self.ndim)) if any(sc != s) else a

        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), self.func(self.s2x(s)), s, reverse_a

    def true_density(self):
        all_int_states = np.int32(list(itertools.product(*[list(range(self.horizon))]*self.ndim)))
        state_mask = np.array([len(self.parent_transitions(s, False)[0]) > 0 or sum(s) == 0
                               for s in all_int_states])
        all_xs = (np.float32(all_int_states) / (self.horizon-1) *
                  (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
        traj_rewards = self.func(all_xs)[state_mask]
        return traj_rewards / traj_rewards.sum(), list(map(tuple,all_int_states[state_mask])), traj_rewards

    def all_possible_states(self):
        # all possible action sequences
        def step_fast(a, s):
            s = s + 0
            s[a] += 1
            return s
        f = lambda a, s: (
            [np.int32(a)] if np.max(s) == self.horizon - 1 else
            [np.int32(a+[self.ndim])]+sum([f(a+[i], step_fast(i, s)) for i in range(self.ndim)], []))
        all_act_seqs = f([], np.zeros(self.ndim, dtype='int32'))
        # all RL states / intermediary nodes
        all_int_states = list(itertools.product(*[list(range(self.horizon))]*self.ndim))
        # Now we need to know for each partial action sequence what
        # the corresponding states are. Here we can just count how
        # many times we moved in each dimension:
        all_traj_states = np.int32([np.bincount(i[:j], minlength=self.ndim+1)[:-1]
                                   for i in all_act_seqs
                                   for j in range(len(i))])
        # all_int_states is ordered, so we can map a trajectory to its
        # index via a sum
        arr_mult = np.int32([self.horizon**(self.ndim-i-1)
                             for i in range(self.ndim)])
        all_traj_states_idx = (
            all_traj_states * arr_mult[None, :]
        ).sum(1)
        # For each partial trajectory, we want the index of which trajectory it belongs to
        all_traj_idxs = [[j]*len(i) for j,i in enumerate(all_act_seqs)]
        # For each partial trajectory, we want the index of which state it leads to
        #all_traj_s_idxs = [[(np.bincount(i, minlength=self.ndim+1)[:-1] * arr_mult).sum()]*len(i)
        #                   for i in all_act_seqs]
        all_traj_s_idxs = [(np.bincount(i, minlength=self.ndim+1)[:-1] * arr_mult).sum()
                           for i in all_act_seqs]
        # Vectorized
        a = torch.cat(list(map(torch.LongTensor, all_act_seqs)))
        u = torch.LongTensor(all_traj_states_idx)
        v1 = torch.cat(list(map(torch.LongTensor, all_traj_idxs)))
        v2 = torch.LongTensor(all_traj_s_idxs)#torch.cat(list(map(torch.LongTensor, all_traj_s_idxs)))
        # With all this we can do an index_add, given
        # pi(all_int_states):
        def compute_all_probs(policy_for_all_states):
            """computes p(x) given pi(a|s) for all s"""
            dev = policy_for_all_states.device
            pi_a_s = torch.log(policy_for_all_states[u, a])
            q = torch.exp(torch.zeros(len(all_act_seqs), device=dev)
                                      .index_add_(0, v1, pi_a_s))
            q_sum = (torch.zeros((all_xs.shape[0],), device=dev)
                     .index_add_(0, v2, q))
            return q_sum[state_mask]
        # some states aren't actually reachable
        state_mask = np.bincount(all_traj_s_idxs, minlength=len(all_int_states)) > 0
        # Let's compute the reward as well
        all_xs = (np.float32(all_int_states) / (self.horizon-1) *
                  (self.xspace[-1] - self.xspace[0]) + self.xspace[0])
        traj_rewards = self.func(all_xs)[state_mask]
        # All the states as the agent sees them:
        all_int_obs = np.float32([self.obs(i) for i in all_int_states])
        print(all_int_obs.shape, a.shape, u.shape, v1.shape, v2.shape)
        return all_int_obs, traj_rewards, all_xs, compute_all_probs

def make_mlp(l, act=nn.LeakyReLU()):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
        for n, (i, o) in enumerate(zip(l, l[1:]))], []))

#@profile
def main(args):
    f = {'default': None,
         'corners': func_corners}[args.func]
    env = GridEnv(args.horizon, args.ndim, func=f)
    envs = [GridEnv(args.horizon, args.ndim, func=f)
            for i in range(args.bufsize)]
    ndim = args.ndim
    policy = make_mlp([args.horizon * ndim] + [args.n_hid] * args.n_layers + [ndim*2])
    policy.to(dev)
    q_target = copy.deepcopy(policy)
    if args.opt == 'adam':
        opt = torch.optim.Adam(policy.parameters(), args.learning_rate,
                               betas=(args.adam_beta1, args.adam_beta2))
    elif args.opt == 'msgd':
        opt = torch.optim.SGD(policy.parameters(), args.learning_rate, momentum=args.momentum)


    if args.compute_l1k1:
        all_inputs, true_r, all_end_states, compute_p_of_x = env.all_possible_states()
        true_density = tf(true_r / true_r.sum())
    else:
        td, end_states, true_r = env.true_density()
        true_density = tf(td)
    print(true_density.shape)

    # The chance of having a random episode
    alpha = args.uniform_sample_prob

    losses, visited, distrib_distances, all_cs = [], [], [], []
    ratios = []
    flow_losses = []
    test_losses = []
    dataset = []
    dataset_distribution = None

    it_range = range(args.n_train_steps+1)
    it_range = tqdm(it_range) if args.progress else it_range
    end_freq = defaultdict(int)
    estimated_density = None
    estimated_densities = []
    bufsize = args.bufsize
    batch = [i.reset() for i in envs]
    dataset = []
    do_MH = False
    do_train = True
    for i in it_range:
        if 0:
            for j in range(bufsize):
                s, r = batch[j]
                logits = policy(tf([s]))
                pi = Categorical(logits=logits)
                a = pi.sample()
                q_xpx = torch.exp(pi.log_prob(a))
                sp, rp, sp_state, reverse_a = envs[j].step(a.item())
                logits_sp = policy(tf([sp]))
                pi_sp = Categorical(logits=logits)
                q_xxp = torch.exp(pi.log_prob(tl([reverse_a])))

                A = (rp * q_xxp) / (r * q_xpx)
                #print(A,  (rp * q_xxp), (r * q_xpx), (rp * q_xxp) / (r * q_xpx))
                if A > torch.rand(1): # Accept
                    batch[j] = (sp, rp)
                    end_freq[tuple(sp_state)] += 1
                if rp > r:
                    dataset.append((tf([s]), a))
        elif do_MH:
            r = np.float32([i[1] for i in batch])
            a = np.random.randint(0, 4, bufsize)
            steps = [envs[j].step(a[j], s=batch[j][2]) for j in range(bufsize)]
            rp = np.float32([i[1] for i in steps])
            A = rp / r
            U = np.random.uniform(0,1,bufsize)
            for j in range(bufsize):
                if A[j] > U[j]: # Accept
                    batch[j] = (None, rp[j], steps[j][2])
                    end_freq[tuple(steps[j][2])] += 1
        else:
            s = torch.cat([tf([i[0]]) for i in batch])
            r = torch.cat([tf([i[1]]) for i in batch])
            logits = policy(s)
            pi = Categorical(logits=logits)
            a = pi.sample()
            q_xpx = torch.exp(pi.log_prob(a))
            steps = [envs[j].step(a[j].item(), s=batch[j][2]) for j in range(bufsize)]
            sp = torch.cat([tf([i[0]]) for i in steps])
            rp = torch.cat([tf([i[1]]) for i in steps])
            logits_sp = policy(sp)
            reverse_a = tl([i[3] for i in steps])
            pi_sp = Categorical(logits=logits_sp)
            q_xxp = torch.exp(pi.log_prob(reverse_a))
            A = (rp * q_xxp) / (r * q_xpx + 1e-6)
            #A = rp / r
            U = torch.rand(bufsize)
            for j in range(bufsize):
                #print(A,  (rp * q_xxp), (r * q_xpx), (rp * q_xxp) / (r * q_xpx))
                if A[j] > U[j]: # Accept
                    batch[j] = (sp[j].numpy(), rp[j].item(), steps[j][2])
                    end_freq[tuple(steps[j][2])] += 1
                if rp[j] > r[j]:
                    dataset.append((s[j].unsqueeze(0), a[j].unsqueeze(0)))





        if do_train:
            if len(dataset) < 512:
                continue
            if not i % 20 and len(dataset) > args.n_dataset_pts:
                dataset = dataset[-args.n_dataset_pts:]
            idxs = np.random.randint(0, len(dataset), args.mbsize)
            s, a = map(torch.cat, zip(*[dataset[i] for i in idxs]))
            logits = policy(s)
            pi = Categorical(logits=logits)
            q_xxp = pi.log_prob(a)
            loss = -q_xxp.mean()
            loss_p = loss - pi.entropy().mean() * 0.1
            loss_p.backward()
            opt.step()
            opt.zero_grad()
            losses.append(loss.item())

        if not i % 2000 and not args.compute_l1k1:
            Z = sum([end_freq[i] for i in end_states])
            estimated_density = tf([end_freq[i] / Z for i in end_states])
            k1 = abs(estimated_density - true_density).mean().item()
            # KL divergence
            kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
            if args.progress:
                print('L1 distance', k1, 'KL', kl, np.mean(losses[-100:]))#, pi.entropy().mean())
                d = np.int32([i[2] for i in steps])
                print(d.min(0), d.max(0))
            distrib_distances.append((k1, kl))
            estimated_densities.append(estimated_density.cpu().numpy())
            end_freq = defaultdict(int)



    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(losses),
         'policy': policy.to('cpu'),
         'est_d': estimated_density.cpu().numpy(),
         'est_ds': estimated_densities,
         'p_dists': distrib_distances,
         'true_d': true_density.cpu().numpy(),
         'training_ratios': np.float32(ratios),
         'learned_cs': all_cs,
         'dataset_d': dataset_distribution,
         'args':args},
        gzip.open(args.save_path, 'wb'))






if __name__ == "__main__":
    args = parser.parse_args()

    tf = lambda x: torch.FloatTensor(x).to(dev)
    tl = lambda x: torch.LongTensor(x).to(dev)
    dev = torch.device(args.device)
    args.progress = True
    torch.set_num_threads(2)
    main(args)
