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
import gpytorch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP

from toy_1d_seq import BinaryTreeEnv, make_mlp

        
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class UCB:
    def __init__(self, model, kappa):
        self.model = model
        self.kappa = kappa
    
    def __call__(self, x):
        t_x = tf(np.array([[x]]))
        output = self.model(t_x)
        mean, std = output.mean, torch.sqrt(output.variance)
        return (mean + self.kappa * std).item()


def main(args):
    # Main Loop
    init_data, all_x, all_y = get_init_data(args, TEST_FUNC)
    init_x, init_y = tf(init_data[:, 0]).unsqueeze(-1), tf(init_data[:, 1])
    exp_name = args.exp_name
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for loss in ["td", "is_xent", "l1", "l2"]:
        distrib_distances = []
        base_path = os.path.join(args.save_path, loss)
        if not os.path.exists(base_path):
            os.mkdir(base_path)

        exp_path = os.path.join(base_path, exp_name)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        dataset = (init_x, init_y)
        model = update_proxy(args, dataset)
        
        args.learning_method = loss

        for i in range(args.num_iter):
            model.eval()
            torch.save(dataset, os.path.join(exp_path, f"dataset-aq-{i}.pth"))
            func = UCB(model, 0.1)
            plot_model(path=os.path.join(exp_path, f'{i}-model.png'), model=model,
                        all_x=all_x, all_y=all_y, train_x=dataset[0], train_y=dataset[1], title=f"Proxy at step: {i}")
            policy = train_generative_model(args, func)
            dataset, metrics = generate_batch(args, policy, dataset, i, exp_path)
            distrib_distances.append(metrics)
            model = update_proxy(args, dataset)
        
            
        pickle.dump(
            {
                'p_dists': distrib_distances,
                'args':args
            },
            gzip.open(os.path.join(exp_path, "result.gz"), 'wb'))


def get_init_data(args, func):
    # Generate initial data to train proxy
    env = BinaryTreeEnv(args.horizon, func=func)
    all_inputs, true_r, all_end_states, compute_p_of_x = env.all_possible_states()
    data = np.dstack((all_end_states, true_r))[0]

    np.random.shuffle(data)
    init_data = data[:args.num_init_points]
    return init_data, tf(all_end_states), tf(true_r)


def generate_batch(args, policy, dataset, it, exp_path):
    # Sample data from trained policy, given dataset. 
    # Currently only naively samples data and adds to dataset, but should ideally also 
    # have a diversity constraint based on existing dataset
    env = BinaryTreeEnv(args.horizon, func=TEST_FUNC)
    all_inputs, true_r, all_end_states, compute_p_of_x = env.all_possible_states()

    true_density = tf(true_r / true_r.sum())
    with torch.no_grad():
        pi_a_s = torch.softmax(policy(tf(all_inputs)), 1)
        estimated_density = compute_p_of_x(pi_a_s)
    # L1 distance
    k1 = abs(estimated_density - true_density).mean().item()
    # KL divergence
    kl = (true_density * torch.log(estimated_density / true_density)).sum().item()

    sampled_x, sampled_y = [], []
    for _ in range(args.num_samples):
        hist, r, log_probs, acts, tstates = run_episode(env, args, policy)
        sampled_x.append([(tstates[-1].reshape((env.horizon, 3)) * env.bitmap_mul).sum()])
        sampled_y.append(r)    
    x, y = dataset

    plot_fn(path=os.path.join(exp_path, f"{it}-aq.png"), all_x=tf(all_end_states), all_y=tf(true_r), 
            train_x=x, train_y=y, batch_x=sampled_x, batch_y=sampled_y, title=f"Points acquired at step {it}")
    x = torch.cat([x, tf(sampled_x)])
    y = torch.cat([y, tf(sampled_y)])
    return (x, y), (k1, kl)


def update_proxy(args, data):
    # Train proxy(GP) on collected data
    train_x, train_y = data
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def train_generative_model(args, func):
    # Train policy using the generative loss, with the given acquisition function
    env = BinaryTreeEnv(args.horizon, func=func)
    policy, q_target, opt, c = initialize_generative_model(args)

    all_inputs, true_r, all_end_states, compute_p_of_x = env.all_possible_states()
    true_density = tf(true_r / true_r.sum())

    # The chance of having a random episode
    alpha = args.uniform_sample_prob
    num_X = all_end_states.shape[0]

    losses, visited, distrib_distances = [], [], []
    ratios = []
    tau = args.bootstrap_tau

    do_queue = args.do_is_queue or args.learning_method == 'is_xent_q'

    queue = []
    queue_thresh = args.queue_thresh
    qids = itertools.count(0) # tie breaker for entries with equal priority

    it_range = range(args.n_train_steps+1)
    it_range = tqdm(it_range) if args.progress else it_range
    for i in it_range:
        batch = []
        trajs = []
        rs = []
        probs = []
        for j in range(args.mbsize):
            if do_queue and len(queue) and np.random.random() < 0.5:
                w, qid, (s, a) = heapq.heappop(queue)
                log_prob = torch.log_softmax(policy(s), 1)[torch.arange(a.shape[0]), a].sum(0, True)
                if -w > 1:
                    heapq.heappush(queue, (-(-w - 1), qid, (s, a)))
                if -w > 1 or np.random.random() < -w:
                    trajs.append(queue_thresh * log_prob)
                    continue
            hist, r, log_probs, acts, tstates = run_episode(env, args, policy)
            batch.extend(hist)
            log_p_theta_x = sum(log_probs)
            smoothed_prob = (alpha / num_X  +
                             (1 - alpha) * torch.exp(log_p_theta_x)).detach()
            rho = r / smoothed_prob
            ratios.append((rho.item(), log_p_theta_x.item()))
            if do_queue and rho > queue_thresh:
                w = rho / queue_thresh - 1
                rho = queue_thresh
                heapq.heappush(queue, (-w.item(), next(qids), (tf(tstates), tl(acts))))
            trajs.append(rho * log_p_theta_x)
            rs.append(tf([r]))
            probs.append(torch.exp(log_p_theta_x))

        loss = get_loss(args.learning_method, batch=batch, trajs=trajs, policy=policy, q_target=q_target, probs=probs, rs=rs, c=c)
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())

        if args.learning_method == "td" and tau > 0:
            for _a,b in zip(policy.parameters(), q_target.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)

    return policy


def initialize_generative_model(args):
    # Initialize params, etc for generative model training
    policy = make_mlp([args.horizon * 3] + [args.n_hid] * args.n_layers + [3])
    policy.to(dev)
    q_target = copy.deepcopy(policy)
    c = None
    if args.learning_method == "l1" or args.learning_method == "l2":
        c = torch.randn(1, requires_grad=True)
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
    
    return policy, q_target, opt, c


def run_episode(env, args, policy):
    # Run a single episode with given policy
    s = env.reset()
    done = False
    is_random = np.random.random() < args.uniform_sample_prob
    log_probs = []
    tstates = []
    acts = []
    hist = []
    while not done:
        tstates.append(s)
        pi = Categorical(logits=policy(tf([s])))
        if is_random:
            a = tl([np.random.randint(3)])
        else:
            a = pi.sample()
        acts.append(a)
        sp, r, done, _ = env.step(a.item())
        hist.append([tf([i]) for i in (s,a,r,sp,done)])
        log_probs.append(pi.log_prob(a))
        s = sp
    return hist, r, log_probs, acts, tstates


def get_loss(loss, **kwargs):
    # Compute loss for updating generative model
    if loss == "td":
        batch, policy, q_target = kwargs['batch'], kwargs['policy'], kwargs['q_target']
        s, a, r, sp, d = map(torch.cat, zip(*batch))
        s = s
        sp = sp
        a = a.long()
        q = policy(s)
        with torch.no_grad(): qp = q_target(sp)
        next_q = qp * (1-d).unsqueeze(1) + d.unsqueeze(1) * (-LOGINF)
        target = torch.logsumexp(torch.cat([torch.log(r).unsqueeze(1), next_q], 1), 1)
        loss = (q[torch.arange(q.shape[0]), a] - target).pow(2).mean()
    elif loss == "is_xent":
        trajs = kwargs['trajs']
        loss = -torch.cat(trajs).mean()
    elif loss == "l1":
        probs, rs, c = kwargs['probs'], kwargs['rs'], kwargs['c']
        loss = torch.abs(torch.cat(probs) - c * torch.cat(rs)).mean()
    elif loss == "l2":
        probs, rs, c = kwargs['probs'], kwargs['rs'], kwargs['c']
        loss = torch.pow(torch.cat(probs) - c * torch.cat(rs), 2).mean()
    return loss


def plot_fn(path, **kwargs):
    fig = plt.figure()
    ax = fig.gca()
    
    if "train_x" in kwargs.keys():
        train_x, train_y = kwargs['train_x'], kwargs['train_y']
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='training points')
    if "batch_x" in kwargs.keys():
        batch_x, batch_y = kwargs['batch_x'], kwargs['batch_y']
        ax.plot(batch_x, batch_y, 'ro', label='acquired points')
    if "all_x" in kwargs.keys():
        all_x, all_y =  kwargs["all_x"], kwargs["all_y"]
        ax.plot(all_x.numpy(), all_y.numpy(), '--', label='true_fn')
    ax.set_title(kwargs["title"])
    ax.legend()
    plt.savefig(path)
    plt.close(fig)
    
    
def plot_model(path, **kwargs):   
    fig = plt.figure()
    ax = fig.gca()
    
    if "all_x" in kwargs.keys():
        all_x, all_y =  kwargs["all_x"], kwargs["all_y"]
        ax.plot(all_x, all_y, '--', label='true_fn')
    if "train_x" in kwargs.keys():
        train_x, train_y = kwargs['train_x'], kwargs['train_y']
        ax.plot(train_x, train_y, 'k*', label='training points')
    if "model" in kwargs.keys():
        model = kwargs["model"]
        out = model(all_x)
        mean = out.mean
        std = torch.sqrt(out.variance)
        with torch.no_grad():
            lower, upper = mean + std, mean - std
            ax.plot(all_x.numpy(), mean.numpy(), 'b', label='Mean')
            ax.fill_between(all_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence')
    ax.set_title(kwargs["title"])
    ax.legend()
    plt.savefig(path)
    plt.close(fig)


parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/', type=str)
parser.add_argument("--exp_name", default='test', type=str)
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
parser.add_argument("--num_init_points", default=64, type=int)
parser.add_argument("--num_iter", default=10, type=int)
parser.add_argument("--num_samples", default=8, type=int)
# This is alpha in the note, smooths the learned distribution into a uniform exploratory one
parser.add_argument("--uniform_sample_prob", default=0.05, type=float)
parser.add_argument("--do_is_queue", action='store_true')
parser.add_argument("--queue_thresh", default=10, type=float)
parser.add_argument("--device", default='cpu', type=str)
parser.add_argument("--progress", action='store_true')

args = parser.parse_args()

# Some general consts
dev = torch.device(args.device)

tf = lambda x: torch.FloatTensor(x).to(dev)
tl = lambda x: torch.LongTensor(x).to(dev)

TEST_FUNC = (lambda x: ((np.cos(x * 50) + 1) * norm.pdf(x * 5)) + 0.01)
LOGINF = torch.tensor(1000).to(dev)

if __name__ == "__main__":
    torch.set_num_threads(1)
    main(args)
