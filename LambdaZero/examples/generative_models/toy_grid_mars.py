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

parser.add_argument("--save_path", default='results/test_flow_corners_3.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='is_xent_q', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
parser.add_argument("--compute_l1k1", default=0, type=int)
parser.add_argument("--horizon", default=20, type=int)
parser.add_argument("--ndim", default=2, type=int)
parser.add_argument("--n_hid", default=64, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=50000, type=int)
parser.add_argument('--func', default='corners')

parser.add_argument("--datasource", default='sample', type=str)
parser.add_argument("--n_dataset_pts", default=50000, type=int)
parser.add_argument("--test_ratio", default=0.05, type=float)

# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.00, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

def func_corners(x):
    return (abs(x) > 0.5).prod(-1) * 0.5 + ((abs(x) < 0.8) * (abs(x) > 0.6)).prod(-1) * 2 + 1e-5

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
        return self.obs()

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

    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0
        if a < self.ndim:
            s[a] += 1

        done = s.max() >= self.horizon - 1 or a == self.ndim
        if _s is None:
            self._state = s
            self._step += 1
        return self.obs(s), 0 if not done else self.func(self.s2x(s)), done, s

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

class weird_act(nn.Module):
    def forward(self, x):
        return abs(x)

def sample_pol_many(policy, envs, mbsize, alpha, end_freq):
    batch = []
    s = tf([i.reset() for i in envs])
    done = [False] * mbsize
    ndim = envs[0].ndim
    while not all(done):
        # Note to self: this is ugly, ugly code
        logits = torch.exp(policy(s))
        pi = logits / (logits.sum(1, keepdim=True) + 1e-10)
        acts = pi.multinomial(1).flatten()
        step = [i.step(a) for i,a in zip([e for d, e in zip(done, envs) if not d], acts)]
        p_a = [envs[0].parent_transitions(sp_state, a == ndim)
               for a, (sp, r, done, sp_state) in zip(acts, step)]
        batch += [[tf(i) for i in (p, a, [r], [sp], [d])]
                  for (p, a), (sp, r, d, _) in zip(p_a, step)]
        c = count(0)
        m = {j:next(c) for j in range(mbsize) if not done[j]}
        done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
        s = tf([i[0] for i in step if not i[2]])
        for (_, _, d, sp) in step:
            if d: end_freq[tuple(sp)] += 1
    return batch

def sample_pol_naive(policy, env, mbsize, alpha, end_freq):
    batch = []
    ndim = env.ndim
    for j in range(mbsize):
        s = env.reset()
        done = False
        is_random = np.random.random() < alpha
        while not done:
            if do_log:
                logits = policy(tf([s]))
                #pi = Categorical(logits=logits)
                logits = torch.exp(logits)
                pi = logits / (logits.sum() + 1e-10)
                a = pi.multinomial(1)
            else:
                q = policy(tf([s]))
                p = q / (q.sum() + 1e-4)
                pi = Categorical(probs=p)
            if is_random:
                a = tl([np.random.randint(ndim+1)])
            else:
                a = a
                #a = pi.sample()
            #if env._step == 0 and a == ndim:
            #    continue # Can't stop in s0
            sp, r, done, sp_state = env.step(a.item())
            parents, actions = env.parent_transitions(sp_state, a == ndim)
            if not len(parents):
                print("Node with no parents??")
                import pdb; pdb.set_trace()
            batch.append([tf(i) for i in (parents, actions,[r],[sp],[done])])
            s = sp
        end_freq[tuple(sp_state)] += 1
    return batch


def main(args):
    f = {'default': None,
         'corners': func_corners}[args.func]
    env = GridEnv(args.horizon, args.ndim, func=f)
    envs = [GridEnv(args.horizon, args.ndim, func=f)
            for i in range(args.mbsize)]
    ndim = args.ndim
    do_log = True
    if do_log:
        policy = make_mlp([args.horizon * ndim] + [args.n_hid] * args.n_layers + [ndim+1])
    else:
        policy = nn.Sequential(
            make_mlp([args.horizon * ndim] + [args.n_hid] * args.n_layers + [ndim+1]),
            weird_act())
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

    #inf = torch.tensor(np.inf).to(dev) # oops, turns out inf * 0 is
                                        # nan rather than 0, so we
                                        # can't actually use inf
    loginf = torch.tensor(1000).to(dev)
    tau = args.bootstrap_tau

    do_sample_online = args.datasource == 'sample'
    do_sample_dataset = args.datasource in ['dataset_uniform', 'dataset_backward',
                                            'dataset_backward_canonical']

    if args.datasource == 'dataset_uniform':
        print("Sampling dataset...")
        dataset, test_xs, test_rs, dataset_distribution = generate_uniform_dataset(
            args, env, true_density, end_states, true_r)
    elif args.datasource == 'dataset_backward' or args.datasource == 'dataset_backward_canonical':
        print("Sampling dataset...")
        dataset, test_xs, test_rs, dataset_distribution = generate_backward_dataset(
            args, env, args.datasource == 'dataset_backward_canonical')

    it_range = range(args.n_train_steps+1)
    it_range = tqdm(it_range) if args.progress else it_range
    end_freq = defaultdict(int)
    estimated_density = None
    estimated_densities = []
    batch = []
    dataset = []
    s, r = env.reset()
    for i in it_range:
        logits = policy(tf([s]))
        pi = Categorical(logits=logits)
        a = pi.sample()
        q_xxp = pi.log_prob(a)
        sp, rp, sp_state, reverse_a = env.step(a.item())
        logits_sp = policy(tf([sp]))
        pi_sp = Categorical(logits=logits)
        q_xpx = pi.log_prob(reverse_a)

        A = torch.minimum(tf(1),
        import pdb; pdb.set_trace()

        if do_sample_online:
            batch = sample_pol_many(policy, envs, args.mbsize, alpha, end_freq)
        elif do_sample_dataset:
            idxs = np.random.randint(0, len(dataset), args.mbsize)
            batch = sum([dataset[i] for i in idxs], [])

        batch_idxs = tl(sum([[i]*len(parents) for i, (parents,_,_,_,_) in enumerate(batch)], []))
        parents, actions, r, sp, done = map(torch.cat, zip(*batch))
        parents_Qsa = policy(parents)[torch.arange(parents.shape[0]), actions.long()]
        if do_log:
            in_flow = torch.log(torch.zeros((sp.shape[0],))
                                .index_add_(0, batch_idxs, torch.exp(parents_Qsa)))
            next_q = policy(sp)
            next_qd = next_q * (1-done).unsqueeze(1) + done.unsqueeze(1) * (-loginf)
            out_flow = torch.logsumexp(torch.cat([torch.log(r)[:, None], next_qd], 1), 1)
        else:
            in_flow = (torch.zeros((sp.shape[0],))
                       .index_add_(0, batch_idxs, parents_Qsa))
            next_q = policy(sp)
            next_qd = next_q * (1-done).unsqueeze(1)
            out_flow = torch.cat([r[:, None], next_qd], 1).sum(1)

        loss = (in_flow - out_flow).pow(2).mean()

        with torch.no_grad():
            term_loss = ((in_flow - out_flow) * done).pow(2).sum() / (done.sum() + 1e-20)
            flow_loss = ((in_flow - out_flow) * (1-done)).pow(2).sum() / ((1-done).sum() + 1e-20)

        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())
        flow_losses.append((term_loss.item(), flow_loss.item()))

        if not i % 50:
            print(list(map(np.mean, zip(*flow_losses[-100:]))))

        if False and do_td and tau > 0:
            for _a,b in zip(policy.parameters(), q_target.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)

        if not i % 250 and args.compute_l1k1:
            with torch.no_grad():
                if do_log:
                    pi_a_s = torch.softmax(policy(tf(all_inputs)), 1)
                else:
                    q = policy(tf(all_inputs))
                    pi_a_s = q / (q.sum(1, keepdim=True) + 1e-9)
                estimated_density = compute_p_of_x(pi_a_s)
            if 0:
                fqs = sorted(end_freq.items())
                Z = sum([i[1] for i in fqs])
                print('>> true, est, emp')
                print(true_density)
                print(np.round(estimated_density.numpy(), 4))
                print(np.round(np.float32([i[1] / Z for i in fqs]), 4))
                print([i[0] for i in fqs])
                print(pi_a_s)
                print(in_flow)
                print(out_flow)
                end_freq = defaultdict(int)


            # L1 distance
            k1 = abs(estimated_density - true_density).mean().item()
            # KL divergence
            kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
            if args.progress:
                print('L1 distance', k1, 'KL', kl, np.mean(losses[-100:]))
            distrib_distances.append((k1, kl))
        if not i % 2000 and not args.compute_l1k1:
            if do_sample_dataset:
                with torch.no_grad():
                    end_freq = defaultdict(int)
                    for i in (tqdm if args.progress else lambda x:x)(range(10000)):
                        s = env.reset()
                        done = False
                        while not done:
                            if do_log:
                                pi = Categorical(logits=policy(tf([s])))
                            else:
                                q = policy(tf([s]))
                                p = q / (q.sum() + 1e-4)
                                pi = Categorical(probs=p)
                            a = pi.sample()
                            sp, r, done, sp_state = env.step(a.item())
                            s = sp
                        end_freq[tuple(sp_state)] += 1
                    test_loss = (policy(test_xs)[:, ndim] - torch.log(test_rs)).pow(2).mean()
                    print(test_loss.item())
                    test_losses.append(test_loss.item())

            fqs = sorted(end_freq.items())
            Z = sum([end_freq[i] for i in end_states])
            estimated_density = tf([end_freq[i] / Z for i in end_states])
            k1 = abs(estimated_density - true_density).mean().item()
            # KL divergence
            kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
            if args.progress:
                print('L1 distance', k1, 'KL', kl, np.mean(losses[-100:]))
            distrib_distances.append((k1, kl))
            estimated_densities.append(estimated_density.cpu().numpy())
            if not do_sample_dataset:
                end_freq = defaultdict(int)



    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(losses),
         'test_losses': np.float32(test_losses),
         'flow_losses': np.float32(flow_losses),
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




def generate_uniform_dataset(args, env, true_density, end_states, true_r):
    dataset = []
    ndim = args.ndim
    eps = 2 / (args.horizon*args.ndim)
    arr_mult = np.int32([args.horizon**(ndim-i-1)
                         for i in range(ndim)])
    test_states = np.random.choice(true_density.shape[0],
                                   int(true_density.shape[0] * 0.1),
                                   replace=False)
    #test_parents, test_actions = zip(*[env.parent_transitions(i) for i in end_states[test_states]])
    test_xs = tf([env.obs(i) for i in np.int32(end_states)[test_states]])
    test_rs = tf(true_r[test_states])
    nrej = 0
    #print(test_states)
    dataset_distribution = np.zeros(true_density.shape[0], np.int32)
    if args.progress:
        prof = tqdm(range(args.n_dataset_pts))
    while len(dataset) < args.n_dataset_pts:
        s = env.reset()
        done = False
        traj = []
        while not done:
            if np.random.uniform() > eps:
                a = np.random.randint(ndim)
            else:
                a = ndim
            sp, r, done, sp_state = env.step(a)
            parents, actions = env.parent_transitions(sp_state, a == ndim)
            traj.append([tf(i) for i in (parents, actions,[r],[sp],[done])])
        state_idx = (sp_state * arr_mult).sum()
        #print([i[-2][0].reshape((2, 30)) for i in traj])
        if state_idx in test_states:
            nrej += 1
            continue
        dataset_distribution[state_idx] += 1
        dataset.append(traj)
        if args.progress:
            prof.update()
    print('done', nrej)
    if args.progress:
        prof.close()
    return dataset, test_xs, test_rs, dataset_distribution


def generate_backward_dataset(args, env, canonical=False):
    td, end_states, true_r = env.true_density()
    true_density = tf(td)
    dataset = []
    ndim = args.ndim
    eps = 2 / (args.horizon*args.ndim)
    arr_mult = np.int32([args.horizon**(ndim-i-1)
                         for i in range(ndim)])
    test_states = np.random.choice(true_density.shape[0],
                                   int(true_density.shape[0] * args.test_ratio),
                                   replace=False)
    test_xs = tf([env.obs(i) for i in np.int32(end_states)[test_states]])
    test_rs = tf(true_r[test_states])
    nrej = 0
    #print(test_states)
    dataset_distribution = np.zeros(true_density.shape[0], np.int32)
    if args.progress:
        prof = tqdm(range(args.n_dataset_pts))
    while len(dataset) < args.n_dataset_pts:
        s0 = np.random.randint(0, args.horizon, args.ndim)
        #print('>', s0)
        state_idx = (s0 * arr_mult).sum()
        if state_idx in test_states:
            #print('skip - test')
            continue
        if len(env.parent_transitions(s0, False)[0]) == 0 and s0.sum() > 0:
            #print('skip - noparent')
            continue
        os0 = env.obs(s0)
        if s0.max() == args.horizon - 1:
            #print(s0, s0.max(), s0.argmax())
            s = s0 + 0
            s[s0.argmax()] -= 1
            traj = [[tf([i]) for i in [env.obs(s), s0.argmax(), env.func(env.s2x(s0)), os0, True]]]
        else:
            traj = [[tf([i]) for i in [os0, ndim, env.func(env.s2x(s0)), os0, True]]]
            s = s0
        while s.sum() > 0:
            #print(s)
            parents, actions = env.parent_transitions(s, False)
            if not len(parents):
                import pdb; pdb.set_trace()
            traj.append([tf(i) for i in (parents, actions, [0], [env.obs(s)], [False])])
            if canonical:
                a = s.argmax()
            else:
                i = np.random.randint(0, len(parents))
                a = actions[i]
            s[a] -= 1
        dataset_distribution[state_idx] += 1
        dataset.append(traj)
        #for i in traj:
        #    print(i)
        if args.progress:
            prof.update()
    if args.progress:
        prof.close()
    return dataset, test_xs, test_rs, dataset_distribution



if __name__ == "__main__":
    args = parser.parse_args()

    tf = lambda x: torch.FloatTensor(x).to(dev)
    tl = lambda x: torch.LongTensor(x).to(dev)
    dev = torch.device(args.device)
    args.progress = True
    torch.set_num_threads(8)
    main(args)
