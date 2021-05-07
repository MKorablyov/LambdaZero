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

parser.add_argument("--save_path", default='results/test_bits_ndim_1.pkl.gz', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--learning_method", default='td', type=str)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--bootstrap_tau", default=0.1, type=float)
parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--num_dims", default=1, type=int)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=10000, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.005, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

def func_cos_N(x):
    return (((np.cos(x * 50) + 1) * norm.pdf(x * 5)) + 0.01).prod(-1)

def func_cos_sin_N(x):
    return (((np.cos(x * 50 - 2) * np.sin(10 * x + 2) + 1) * norm.pdf(x * 5, loc=2, scale=2)) + 0.01).prod(-1)

def func_corners_tree(x):
    pass

funcs = {'cos_N': func_cos_N,
         'cos_sin_N': func_cos_sin_N,
         'corners_tree': func_corners_tree,
}


class BinaryTreeEnv:

    def __init__(self, horizon, ndim=1, xrange=[-1, 1], func=None):
        self.horizon = horizon
        self.center = sum(xrange)/2
        self.width = xrange[1] - xrange[0]
        self.func = func_cos_N if func is None else func
        self.bitmap = np.float32([1/2**(i+1) for i in range(horizon)])
        self.bitmap_mul = np.zeros((horizon, 1, 3))
        self.bitmap_mul[:, 0, 0] = self.bitmap
        self.bitmap_mul[:, 0, 1] = -self.bitmap
        self.ndim = ndim
        self.xrange = xrange
        self._true_dist = None

    def obs(self, s=None):
        s = self._state if s is None else s
        z = np.zeros((self.horizon, self.ndim, 3), dtype=np.float32)
        hidx = np.concatenate([np.arange(len(i)) for i in s])
        didx = np.concatenate([np.int32([j] * len(i)) for j, i in enumerate(s)])
        bidx = np.concatenate(list(map(np.int32, s)))
        z[:, :, 2] = 1
        z[hidx, didx, bidx] = 1
        z[hidx, didx, 2] = 0
        return z.flatten()

    def s2x(self, s):
        return (self.obs(s).reshape((self.horizon, self.ndim, 3)) * self.bitmap_mul).sum((0, 2))

    def reset(self):
        self._state = self.get_reset_state()
        self._step = 0
        return self.obs()

    def get_reset_state(self):
        return [[] for i in range(self.ndim)]

    def compute_rewards(self, x):
        if hasattr(self.func, 'many'):
            return self.func.many(x)
        return self.func(x) # [self.func(i) for i in x]

    @property
    def true_dist(self):
        if self._true_dist is None:
            cube = np.stack(np.meshgrid(*[
                np.linspace(*self.xrange, 2 * 2 ** self.horizon+1)[1:-1]]*self.ndim))
            cube = cube.transpose(*range(1, self.ndim+1), 0)
            r = (self.func.many(torch.tensor(cube).float().reshape(-1, self.ndim))
                 .numpy().reshape(cube.shape[:-1])
                 if hasattr(self.func, 'many') else self.func(cube))
            self._true_dist = r / r.sum()
        return self._true_dist

    def compute_empirical_distribution_error(self, visited_states):
        """return L1 and KL divergence for visited_states a list of xs

        the xs are assumed to be returned from BinaryTreeEnv.step(..., return_x=True)
        """
        assert len(visited_states)
        eps = 1e-3 * 2 ** -self.horizon
        s = np.float32(visited_states)
        x2idx = (s - self.xrange[0]) * 2 ** self.horizon - 1
        hist, bins = np.histogramdd(x2idx, [np.arange(2*2**self.horizon)]*self.ndim)
        empirical = hist / hist.sum()
        L1 = abs(empirical - self.true_dist).mean()
        KL = -(self.true_dist * np.log((empirical + eps) / self.true_dist)).sum()
        return L1, KL

    def step(self, a, s=None, return_x=False):
        """
        steps the env:
         - a: the action, int in [0, ndim * 2 + 1]
         - s: the integer position state (or self._state if s is None)
         - return_x: if True, returns x instead of self.func(x) (i.e. R(x)), useful for batching
        Cannot step the environment with a if env.allowed_acts(s)[a] is 0
        """
        _s = s
        s = self._state if s is None else s
        dim = a // 2 # The dimension we're adding to
        bit = a % 2 # The bit we're adding
        if dim < self.ndim:
            if len(s[dim]) >= self.horizon:
                raise ValueError(f'action `{a}` is not legal: s={s}, H={self.horizon}')
            s[dim].append(bit)
        done = dim == self.ndim or min(map(len, s)) >= self.horizon
        if _s is None:
            self._state = s
        o = self.obs(s)
        if done:
            x = self.s2x(s)
            r = x if return_x else self.func(x)
        else:
            r = 0
        return o, r, done, s

    def all_possible_states(self):
        if self.ndim == 1:
            return self.all_possible_states_1d()
        raise NotImplementedError()

    def allowed_acts(self, s=None):
        s = self._state if s is None else s
        lens = list(map(len,s))
        cumlens = np.cumsum(lens[::-1])[::-1]
        return np.float32(sum([[1,1] if lens[i] < self.horizon and cumlens[i] == lens[i] else [0,0]
                               for i in range(self.ndim)], []) + [1])

    def all_possible_states_1d(self):
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
        all_xs = (np.float32(all_bitseqs) * self.bitmap_mul[None, :, 0]).sum((1,2))
        all_bitseqs, all_xs, all_act_seqs = (
            zip(*sorted(zip(all_bitseqs, all_xs, all_act_seqs), key=lambda x:x[1])))
        all_bitseqs = np.float32(all_bitseqs)
        all_xs = np.float32(all_xs)
        smap = dict(zip(all_xs, range(len(all_xs))))
        # This is to help us compute the index_add in compute_all_probs
        # to compute p_theta(x) for all xs.
        # The flattened sequence of state index in each trajectory
        all_idx_trajs = [smap[(self.bitmap_mul[:k, 0]*i[:k]).sum()]
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

    def construct_validation_set(self, n, seed=142857):
        '''
        Constructs a set of trajectories used to test a policy
        '''
        if n >= (2 * (2 ** self.horizon)-1) ** self.ndim:
            raise ValueError("Trying to sample more unique states then there are states")
        rng = np.random.RandomState(seed)
        seen = set()
        # Since we're in a binary search tree, we want each level to
        # be sampled with probability 1/(2^(H-i)), i.e. we want the
        # root to be chosen with probability 1/(2^H) and a leaf to be
        # chosen with probability 1/2
        p = np.float32([1/2**i for i in range(self.horizon+1)][::-1])
        p = p / p.sum() # We still need to normalize this because
                        # sum_i^N 1/2^i == 1 iff N -> infty
        all_observations = []
        all_actions = []
        all_allowed_acts = []
        all_traj_idxs = []
        all_probs = []
        while len(seen) < n:
            # Sample a state by sampling the bit sequences
            sp = [rng.randint(0, 2, rng.multinomial(1, p).argmax())
                  for i in range(self.ndim)]
            # The corresponding hypercube grid index
            idx = np.int32((self.s2x(sp) + 1) * 2 ** self.horizon)-1
            if tuple(idx) in seen:
                continue
            seen.add(tuple(idx))
            # Compute the list of actions and observations that lead
            # to that state
            sij = lambda i,j: sp[:i] + [sp[i][:j]] + [[]] * (self.ndim - i - 1)
            actions = [sp[dim][j] + dim * 2
                       for dim in range(self.ndim)
                       for j in range(len(sp[dim]))]
            allowed_acts = [self.allowed_acts(sij(dim, j))
                            for dim in range(self.ndim)
                            for j in range(len(sp[dim]))]
            observs = [self.obs(sij(dim, j))
                       for dim in range(self.ndim)
                       for j in range(len(sp[dim]))]
            # Terminal state
            actions.append(self.ndim * 2)
            allowed_acts.append(self.allowed_acts(sp))
            observs.append(self.obs(sp))
            all_traj_idxs += [len(seen) - 1] * len(actions)
            all_observations += observs
            all_actions += actions
            all_allowed_acts += allowed_acts
            all_probs.append(self.true_dist[tuple(idx)])

        self._valid_observations = torch.tensor(all_observations)
        self._valid_actions = torch.LongTensor(all_actions)
        self._valid_allowed_acts = torch.tensor(all_allowed_acts)
        self._valid_traj_idxs = torch.LongTensor(all_traj_idxs)
        self._valid_probs = torch.tensor(all_probs)

    def compute_validation_error(self, policy):
        '''Compute the validation L1 and KL errors.

        Assumes that policy(x) returns the unnormalized logits of a
        categorical distribution
        '''
        with torch.no_grad():
            logits = policy(self._valid_observations)
        logits = logits * self._valid_allowed_acts - 100 * (1-self._valid_allowed_acts)
        pi_a_s = Categorical(logits=logits).log_prob(self._valid_actions)

        p_tau = torch.exp(torch.zeros(self._valid_probs.shape[0])
                          .index_add_(0, self._valid_traj_idxs, pi_a_s))
        l1 = abs(p_tau - self._valid_probs).mean()
        kl = -(self._valid_probs * torch.log(p_tau / self._valid_probs)).sum()
        return l1.item(), kl.item()

def make_mlp(l, act=nn.LeakyReLU()):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*sum(
        [[nn.Linear(i, o)] + ([act] if n < len(l)-2 else [])
        for n, (i, o) in enumerate(zip(l, l[1:]))], []))

def main(args):
    env = BinaryTreeEnv(args.horizon, ndim=args.num_dims)
    dev = torch.device(args.device)
    policy = make_mlp([args.horizon * args.num_dims * 3] +
                      [args.n_hid] * args.n_layers +
                      [args.num_dims * 2 + 1])
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
                    trajs.append((queue_thresh * log_prob, log_prob, None))
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
                    a = tl([np.random.randint(args.num_dims*2+1)])
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
            r = tf([i[2] for i in trajs])
            loss = torch.abs(torch.exp(log_p_theta_x) - c * r).mean()
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
