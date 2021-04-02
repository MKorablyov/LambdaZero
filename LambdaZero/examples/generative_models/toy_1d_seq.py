import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/test_bits_q_1.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='is_xent_q', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.05, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')


class BinaryTreeEnv:

    def __init__(self, horizon, ndim=1, xrange=[-1, 1], func=None):
        self.horizon = horizon
        self.center = sum(xrange)/2
        self.width = xrange[1] - xrange[0]
        self.func = (
            (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)) + 0.01)
            if func is None else func)
        self.bitmap = np.float32([1/2**(i+1) for i in range(horizon)])
        self.bitmap_mul = np.zeros((horizon, 3))
        self.bitmap_mul[:, 0] = self.bitmap
        self.bitmap_mul[:, 1] = -self.bitmap

        if ndim > 1:
            raise NotImplementedError()

    def obs(self, s=None):
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon, 3), dtype=np.float32)
        z[np.arange(len(s)), s] = 1
        z[len(s):, 2] = 1
        return z.flatten()

    def s2x(self, s):
        return (self.obs(s).reshape((self.horizon, 3)) * self.bitmap_mul).sum()

    def reset(self):
        self._state = []
        self._step = 0
        return self.obs()

    def step(self, a, s=None):
        _s = s
        s = self._state if s is None else s
        if a == 0 or a == 1:
            s.append(a)

        done = len(s) >= self.horizon or a == 2
        if _s is None:
            self._state = s
        return self.obs(), 0 if not done else self.func(self.s2x(s)), done, s

    def all_possible_states(self):
        # expanded bit sequences, [0,1,end] turns into [[1,0,0],[0,1,0],[0,0,1],[0,0,1]...]
        all_bitseqs = sum([
            list(list(i) + [[0,0,1]]*(self.horizon-L)
                 for i in itertools.product(*[[[1,0,0],[0,1,0]]]*L))
            for L in range(self.horizon+1)], []
        )
        # all action sequences
        all_act_seqs = sum([
            list(list(i) + ([2] if L < self.horizon else [])
                 for i in itertools.product(*[[0,1]]*L))
            for L in range(self.horizon+1)], []
        )
        # xs corresponding to bit sequences
        all_xs = (np.float32(all_bitseqs) * self.bitmap_mul[None]).sum((1,2))
        all_bitseqs, all_xs, all_act_seqs = (
            zip(*sorted(zip(all_bitseqs, all_xs, all_act_seqs), key=lambda x:x[1])))
        all_bitseqs = np.float32(all_bitseqs)
        all_xs = np.float32(all_xs)
        smap = dict(zip(all_xs, range(len(all_xs))))
        # This is to help us compute the index_add in compute_all_probs
        # to compute p_theta(x) for all xs.
        # The flattened sequence of state index in each trajectory
        all_idx_trajs = [smap[(self.bitmap_mul[:k]*i[:k]).sum()]
                         for i, j in zip(all_bitseqs, all_act_seqs)
                         for k in range(len(j))]
        # The index of each trajectory
        all_traj_idxs = [[j]*len(i) for j,i in enumerate(all_act_seqs)]
        # Vectorized
        a = torch.cat(list(map(torch.LongTensor, all_act_seqs)))
        u = torch.LongTensor(all_idx_trajs)
        v = torch.cat(list(map(torch.LongTensor, all_traj_idxs)))
        def compute_all_probs(policy_for_all_states):
            """computes p(x) given pi(a|s) for all s"""
            dev = policy_for_all_states.device
            pi_a_s = torch.log(policy_for_all_states[u, a])
            return torch.exp(
                torch.zeros((all_xs.shape[0],), device=dev)
                .index_add_(0, v.to(dev), pi_a_s))
        traj_rewards = np.float32([self.func(i) for i in all_xs])
        return all_bitseqs.reshape((-1, self.horizon*3)), traj_rewards, all_xs, compute_all_probs

def make_mlp(l, act=nn.LeakyReLU()):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
        for n, (i, o) in enumerate(zip(l, l[1:]))], []))

def main(args):
    env = BinaryTreeEnv(args.horizon)
    dev = torch.device(args.device)
    policy = make_mlp([args.horizon * 3] + [args.n_hid] * args.n_layers + [3])
    policy.to(dev)
    q_target = copy.deepcopy(policy)
    if args.learning_method == "l1" or args.learning_method == "l2":
        c = torch.zeros(1, requires_grad=True)
        if args.opt == 'adam':
            opt = torch.optim.Adam(list(policy.parameters()) + [c], args.learning_rate,
                                betas=(args.adam_beta1, args.adam_beta2))
        elif args.opt == 'msgd':
            opt = torch.optim.SGD(list(policy.parameters()) + [c], args.learning_rate, momentum=args.momentum)
    else:
        if args.opt == 'adam':
            opt = torch.optim.Adam(policy.parameters(), args.learning_rate,
                                betas=(args.adam_beta1, args.adam_beta2))
        elif args.opt == 'msgd':
            opt = torch.optim.SGD(policy.parameters(), args.learning_rate, momentum=args.momentum)

    tf = lambda x: torch.FloatTensor(x).to(dev)
    tl = lambda x: torch.LongTensor(x).to(dev)

    all_inputs, true_r, all_end_states, compute_p_of_x = env.all_possible_states()
    true_density = tf(true_r / true_r.sum())


    # The chance of having a random episode
    alpha = args.uniform_sample_prob
    num_X = all_end_states.shape[0]

    losses, visited, distrib_distances, all_cs = [], [], [], []
    ratios = []

    #inf = torch.tensor(np.inf).to(dev) # oops, turns out inf * 0 is
                                        # nan rather than 0, so we
                                        # can't actually use inf
    loginf = torch.tensor(1000).to(dev)
    tau = args.bootstrap_tau

    do_td = args.learning_method == 'td'
    do_isxent = 'is_xent' in args.learning_method
    do_queue = args.do_is_queue or args.learning_method == 'is_xent_q'
    do_l1 = args.learning_method == 'l1'
    do_l2 = args.learning_method == 'l2'

    queue = []
    queue_thresh = args.queue_thresh
    qids = itertools.count(0) # tie breaker for entries with equal priority

    it_range = range(args.n_train_steps+1)
    it_range = tqdm(it_range) if args.progress else it_range
    for i in it_range:
        batch = []
        trajs = []
        batch_log_probs = []
        for j in range(args.mbsize):
            if do_queue and len(queue) and np.random.random() < 0.5:
                w, qid, (s, a) = heapq.heappop(queue)
                log_prob = torch.log_softmax(policy(s), 1)[torch.arange(a.shape[0]), a].sum(0, True)
                if -w > 1:
                    heapq.heappush(queue, (-(-w - 1), qid, (s, a)))
                if -w > 1 or np.random.random() < -w:
                    trajs.append(queue_thresh * log_prob)
                    continue
            s = env.reset()
            done = False
            is_random = np.random.random() < alpha
            log_probs = []
            tstates = []
            acts = []
            while not done:
                tstates.append(s)
                pi = Categorical(logits=policy(tf([s])))
                if is_random:
                    a = tl([np.random.randint(3)])
                else:
                    a = pi.sample()
                acts.append(a)
                sp, r, done, _ = env.step(a.item())
                batch.append([tf([i]) for i in (s,a,r,sp,done)])
                log_probs.append(pi.log_prob(a))
                s = sp
            log_p_theta_x = sum(log_probs)
            smoothed_prob = (alpha / num_X  +
                             (1 - alpha) * torch.exp(log_p_theta_x)).detach()
            rho = r / smoothed_prob
            ratios.append((rho.item(), log_p_theta_x.item()))
            if do_queue and rho > queue_thresh:
                w = rho / queue_thresh - 1
                rho = queue_thresh
                heapq.heappush(queue, (-w.item(), next(qids), (tf(tstates), tl(acts))))
            trajs.append((rho, log_p_theta_x, r))


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
            loss = -torch.cat([i[0] * i[1] for i in trajs]).mean()
        elif do_l1:
            log_p_theta_x = torch.cat([i[1] for i in trajs])
            r = tf([i[2] for i in trajs])            loss = torch.abs(torch.exp(log_p_theta_x) - c * r).mean()
        elif do_l2:
            log_p_theta_x = torch.cat([i[1] for i in trajs])
            r = tf([i[2] for i in trajs])
            loss = torch.pow(torch.exp(log_p_theta_x) - c * r, 2).mean()

        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())

        if do_td and tau > 0:
            for _a,b in zip(policy.parameters(), q_target.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)

        if not i % 50:
            with torch.no_grad():
                pi_a_s = torch.softmax(policy(tf(all_inputs)), 1)
                estimated_density = compute_p_of_x(pi_a_s)
            # L1 distance
            k1 = abs(estimated_density - true_density).mean().item()
            # KL divergence
            kl = (true_density * torch.log(estimated_density / true_density)).sum().item()
            if do_l1 or do_l2:
                all_cs.append(c.item())
            if args.progress:
                print('L1 distance', k1, 'KL', kl, np.mean(losses[-100:]),
                      len(queue) if do_queue else '')
            distrib_distances.append((k1, kl))

    root = os.path.split(args.save_path)[0]
    os.makedirs(root, exist_ok=True)
    pickle.dump(
        {'losses': np.float32(losses),
         'policy': policy.to('cpu'),
         'est_d': estimated_density.cpu().numpy(),
         'p_dists': distrib_distances,
         'true_d': true_density.cpu().numpy(),
         'training_ratios': np.float32(ratios),
         'learned_cs': all_cs,
         'args':args},
        gzip.open(args.save_path, 'wb'))



if __name__ == "__main__":
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
