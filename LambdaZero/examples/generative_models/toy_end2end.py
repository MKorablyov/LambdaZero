import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle

import numpy as np
from scipy.stats import norm
from scipy.spatial import distance_matrix
from tqdm import tqdm
import torch
import gpytorch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from torch.utils.data import TensorDataset, DataLoader

from toy_1d_seq import BinaryTreeEnv, make_mlp

class UCB:
    def __init__(self, model, kappa):
        self.model = model
        self.kappa = kappa

    def __call__(self, x):
        t_x = tf(np.array([[x]]))
        output = self.model(t_x)
        mean, std = output.mean, torch.sqrt(output.variance)
        return (mean + self.kappa * std).item()
        t_x = tf(np.array([[x]]))

    def many(self, x):
        output = self.model(x)
        mean, std = output.mean, torch.sqrt(output.variance)
        return (mean + self.kappa * std)


def main(args):
    # Main Loop
    init_data, val_data, all_end_states, true_r, compute_p_of_x, all_inputs, env = get_init_data(args, TEST_FUNC)
    all_x, all_y = tf(all_end_states), tf(true_r)
    init_x, init_y = tf(init_data[:, 0]).unsqueeze(-1), tf(init_data[:, 1])
    val_data = (tfc(val_data[0]), tfc(val_data[1]))
    exp_name = args.exp_name
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    for loss in ["td", "is_xent", "l1", "l2"]:
        distrib_distances = []
        reward = []
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
            plot_model(path=os.path.join(exp_path, f'model-{i}.png'), model=model,
                        all_x=all_x, all_y=all_y, train_x=dataset[0], train_y=dataset[1], title=f"Proxy at step: {i}")
            policy, train_loss, val_loss = train_generative_model(args, func, val_data, compute_p_of_x)
            # plot_losses(args, train_loss, val_loss, os.path.join(exp_path, f'losses-{i}.png'))
            new_dataset, metrics = generate_batch(args, policy, dataset, i, exp_path,
                                                env, all_inputs, true_r, all_end_states, compute_p_of_x)
            reward.append(diverse_topk_mean_reward(args, dataset, new_dataset))
            dataset = new_dataset
            distrib_distances.append(metrics)
            model = update_proxy(args, dataset)

        pickle.dump(
            {
                'p_dists': distrib_distances,
                'outer_loop_reward': reward,
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
    idx = np.random.randint(0, len(true_r), size=args.num_val_points)
    val_data = (np.array(all_inputs)[idx], np.array(true_r)[idx])
    return init_data, val_data, all_end_states, true_r, compute_p_of_x, all_inputs, env


def get_network_output(args, network, inputs, mean_std=False):
    dataset = TensorDataset(inputs)
    dataloader = DataLoader(dataset, args.inf_batch_size, num_workers=0, shuffle=False)
    if not mean_std:
        outputs = []
        for batch in dataloader:
            outputs.append(network(batch[0]))
        return torch.cat(outputs, dim=0)
    else:
        mean = []
        std = []
        for batch in dataloader:
            out = network(batch[0])
            mean.append(out.mean.cpu())
            std.append(torch.sqrt(out.variance).cpu())
        return torch.cat(mean, dim=0), torch.cat(std, dim=0)


def generate_batch(args, policy, dataset, it, exp_path, env, all_inputs, true_r, all_end_states, compute_p_of_x):
    # Sample data from trained policy, given dataset.
    # Currently only naively samples data and adds to dataset, but should ideally also
    # have a diversity constraint based on existing dataset
    true_density = tf(true_r / true_r.sum())
    with torch.no_grad():
        pi_a_s = torch.softmax(get_network_output(args, policy, tf(all_inputs)), 1)
        estimated_density = compute_p_of_x(pi_a_s)
    plot_densities(args, np.array(all_end_states), true_density.cpu().numpy(), estimated_density.cpu().numpy(), path=os.path.join(exp_path, f"est_dens-{it}.png"))
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

    plot_fn(path=os.path.join(exp_path, f"aq-{it}.png"), all_x=tf(all_end_states), all_y=tf(true_r),
            train_x=x, train_y=y, batch_x=sampled_x, batch_y=sampled_y, title=f"Points acquired at step {it}")
    x = torch.cat([x, tf(sampled_x)])
    y = torch.cat([y, tf(sampled_y)])
    return (x, y), (k1, kl)


def update_proxy(args, data):
    # Train proxy(GP) on collected data
    train_x, train_y = data
    model = SingleTaskGP(train_x, train_y.unsqueeze(-1),
                         covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5, lengthscale_prior=gpytorch.priors.GammaPrior(0.5, 2.5))))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model


def train_generative_model(args, func, val_data, compute_p_of_x):
    # Train policy using the generative loss, with the given acquisition function
    env = BinaryTreeEnv(args.horizon, func=func)
    policy, q_target, opt, c = initialize_generative_model(args)

    # The chance of having a random episode
    alpha = args.uniform_sample_prob
    num_X = 2 ** args.horizon - 1
    a_over_X = alpha / num_X

    losses, visited, distrib_distances = [], [], []
    tau = args.bootstrap_tau

    do_queue = args.do_is_queue or args.learning_method == 'is_xent_q'
    if do_queue:
        # I've removed this due to the batching update and since we're
        # not using this method anyways. I will reimplement if
        # needed. -Emmanuel
        raise NotImplementedError()

    val_losses = []
    for i in tqdm(range(args.n_train_steps+1), disable=not args.progress):
        transitions, rewards, log_p_theta_x, all_obs = run_episodes(
            env, args, policy, args.mbsize)
        loss = get_batch_loss(args.learning_method,
                              transitions=transitions, rewards=rewards, log_p_theta_x=log_p_theta_x,
                              policy=policy, q_target=q_target, c=c,
                              alpha=alpha, num_X=num_X)
        loss.backward()
        opt.step()
        opt.zero_grad()

        losses.append(loss.item())

        if args.learning_method == "td" and tau > 0:
            for _a,b in zip(policy.parameters(), q_target.parameters()):
                b.data.mul_(1-tau).add_(tau*_a)
        # TODO: Add prob density computation for set of points
        # if i % args.num_val_iters:
        #     val_loss = get_val_loss(args, policy, val_data, compute_p_of_x)
        #     val_losses.append(val_loss.item())

    return policy, losses, val_losses


def get_val_loss(args, policy, val_data, compute_p_of_x):
    with torch.no_grad():
        pi_a_s = torch.softmax(policy(val_data[0].to(dev)), 1)
        import pdb; pdb.set_trace();
        estimated_density = compute_p_of_x(pi_a_s)
    loss_fn =  lambda c: torch.sum(estimated_density - c * val_data[1])


def diverse_topk_mean_reward(args, d_prev, d):
    topk_new, new_indices = torch.topk(d[1], k=args.reward_topk)
    topk_old, old_indices = torch.topk(d_prev[1], k=args.reward_topk)
    new_reward = topk_new.mean() + args.reward_lambda * get_pairwise_distances(d[0][new_indices].cpu().numpy())
    old_reward = topk_old.mean() + args.reward_lambda * get_pairwise_distances(d_prev[0][old_indices].cpu().numpy())
    return (new_reward - old_reward).item()


def get_pairwise_distances(arr):
    return np.sum(np.tril(distance_matrix(arr, arr))) * 2 / (arr.shape[0] * (arr.shape[0] - 1 ))


def initialize_generative_model(args, init_params={}):
    # Initialize params, etc for generative model training
    # init_params will be a dict containing the parameters for each model.

    policy = make_mlp([args.horizon * 3] + [args.n_hid] * args.n_layers + [3])
    if 'policy' in init_params.keys():
        policy.load_state_dict(init_params['policy'])
    policy.to(dev)
    q_target = copy.deepcopy(policy)
    if 'q_target' in init_params.keys():
        q_target.load_state_dict(init_params['q_target'])
    c = None
    if args.learning_method == "l1" or args.learning_method == "l2":
        c = torch.randn(1, requires_grad=True)
        if 'c' in init_params.keys():
            c = init_params['c']
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
    if 'opt' in init_params.keys():
        opt.load_state_dict(init_params['opt'])
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
        try:
            pi = Categorical(logits=policy(tf([s])))
        except Exception as e:
            print(policy(tf([s])), e)
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


def run_episodes(env, args, policy, mbsize):
    # Run mbsize episodes with the given policy
    s = [env.get_reset_state() for i in range(mbsize)]
    done = [False] * mbsize
    running = tl(range(mbsize))
    is_random = torch.tensor(np.random.random(mbsize) < args.uniform_sample_prob)
    n_random = is_random.sum().item()
    transitions = [] # list of (s,a,r,sp,done) batch-tuples
    log_probs = torch.zeros(mbsize) # the trajectory log probabilities
    all_observations = []
    hist = []

    observations = tf(list(map(env.obs, s)))
    rewards_to_compute = [] # keep a list of rewards to ask the proxy
                            # for so we can benefit from batching
    while len(running):
        all_observations.append(observations)
        pi = Categorical(logits=policy(observations))
        a = pi.sample()
        if n_random:
            a[is_random] = tl(np.random.randint(0, 3, n_random))
        obs_sp, xs, done, sp = zip(*[env.step(ai.item(), si, return_x=True) for ai, si in zip(a, s)])
        obs_sp = tf(obs_sp)
        transitions.append([observations, a, torch.zeros(len(xs)), obs_sp, tf(done)])
        for i, d in enumerate(done):
            if d: rewards_to_compute.append((len(transitions)-1, i, xs[i]))
        log_probs = log_probs.index_add(0, tl(running), pi.log_prob(a))
        if any(done):
            mask = (1-tl(done)).bool()
            running = running[mask]
            observations = obs_sp[mask]
            is_random = is_random[mask]
            n_random = is_random.sum().item()
            s = [i for i, d in zip(sp, done) if not d]
        else:
            s = sp
            observations = obs_sp
    # Compute rewards and reinsert them into the transitions
    xs = tf([i[2] for i in rewards_to_compute])
    rs = env.compute_rewards(xs)
    for (i, j, _), r in zip(rewards_to_compute, rs):
        transitions[i][2][j] = r
    # Stack transition tensors
    batched_transitions = [torch.cat(i, 0) for i in zip(*transitions)]
    return batched_transitions, rs, log_probs, all_observations


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


def get_batch_loss(loss, **kwargs):
    # Compute loss for updating generative model
    if loss == "td":
        policy, q_target = kwargs['policy'], kwargs['q_target']
        s, a, r, sp, d = kwargs['transitions']
        s = s
        sp = sp
        a = a.long()
        q = policy(s)
        with torch.no_grad(): qp = q_target(sp)
        next_q = qp * (1-d).unsqueeze(1) + d.unsqueeze(1) * (-LOGINF)
        target = torch.logsumexp(torch.cat([torch.log(r).unsqueeze(1), next_q], 1), 1)
        loss = (q[torch.arange(q.shape[0]), a] - target).pow(2).mean()
    elif loss == "is_xent":
        r, log_p_theta_x = kwargs['rewards'], kwargs['log_p_theta_x']
        smoothed_prob = (kwargs['alpha'] / kwargs['num_X']  +
                         (1 - kwargs['alpha']) * torch.exp(log_p_theta_x)).detach()
        rho = r / smoothed_prob
        loss = -(rho * log_p_theta_x).mean()
    elif loss == "l1":
        log_p_theta_x, rewards, c = kwargs['log_p_theta_x'], kwargs['rewards'], kwargs['c']
        loss = torch.abs(torch.exp(log_p_theta_x) - c * rewards).mean()
    elif loss == "l2":
        log_p_theta_x, rewards, c = kwargs['log_p_theta_x'], kwargs['rewards'], kwargs['c']
        loss = torch.pow(torch.exp(log_p_theta_x) - c * rewards, 2).mean()
    return loss



def plot_losses(args, train_losses, val_losses, path):
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(train_losses, label='train_loss')
    ax.plot(np.arange(len(val_losses)) * args.num_val_iters, val_losses, label='train_loss')
    ax.set_title('Losses')
    ax.legend()
    plt.savefig(path)
    plt.close(fig)


def plot_fn(path, **kwargs):
    fig = plt.figure()
    ax = fig.gca()

    if "train_x" in kwargs.keys():
        train_x, train_y = kwargs['train_x'], kwargs['train_y']
        train_x, train_y = train_x.cpu(), train_y.cpu()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='training points')
        train_x, train_y = train_x.to(dev), train_y.to(dev)
    if "batch_x" in kwargs.keys():
        batch_x, batch_y = kwargs['batch_x'], kwargs['batch_y']
        ax.plot(batch_x, batch_y, 'ro', label='acquired points')
    if "all_x" in kwargs.keys():
        all_x, all_y =  kwargs["all_x"], kwargs["all_y"]
        all_x, all_y = all_x.cpu(), all_y.cpu()
        ax.plot(all_x.numpy(), all_y.numpy(), '--', label='true_fn')
        all_x, all_y = all_x.to(dev), all_y.to(dev)
    ax.set_title(kwargs["title"])
    ax.legend()
    plt.savefig(path)
    plt.close(fig)

def plot_densities(args, x, true_density, estimated_density, path):
    fig = plt.figure()
    ax = fig.gca()

    ax.plot(x, true_density, '--', label='true_density')
    ax.plot(x, estimated_density, label='estimated density')

    ax.set_title("Estimated density")
    ax.legend()
    plt.savefig(path)
    plt.close(fig)

def plot_model(path, **kwargs):
    fig = plt.figure()
    ax = fig.gca()

    if "all_x" in kwargs.keys():
        all_x, all_y =  kwargs["all_x"], kwargs["all_y"]
        all_x, all_y = all_x.cpu(), all_y.cpu()
        ax.plot(all_x.numpy(), all_y.numpy(), '--', label='true_fn')
        all_x, all_y = all_x.to(dev), all_y.to(dev)
    if "train_x" in kwargs.keys():
        train_x, train_y = kwargs['train_x'], kwargs['train_y']
        train_x, train_y = train_x.cpu(), train_y.cpu()
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='training points')
        train_x, train_y = train_x.to(dev), train_y.to(dev)
    if "model" in kwargs.keys():
        model = kwargs["model"]
        mean, std = get_network_output(args, model, all_x, mean_std=True)
        all_x = all_x.cpu()
        with torch.no_grad():
            lower, upper = mean + std, mean - std
            ax.plot(all_x.numpy(), mean.numpy(), 'b', label='Mean')
            ax.fill_between(all_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence')
        all_x = all_x.to(dev)
    ax.set_title(kwargs["title"])
    ax.legend()
    plt.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    torch.set_num_threads(1)

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", default='results/', type=str)
    parser.add_argument("--exp_name", default='test', type=str)
    parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
    parser.add_argument("--learning_method", default='is_xent_q', type=str)
    parser.add_argument("--opt", default='adam', type=str)
    parser.add_argument("--adam_beta1", default=0.9, type=float)
    parser.add_argument("--adam_beta2", default=0.999, type=float)
    parser.add_argument("--reward_lambda", default=0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--bootstrap_tau", default=0.1, type=float)
    parser.add_argument("--mbsize", default=8, help="Minibatch size", type=int)
    parser.add_argument("--horizon", default=8, type=int)
    parser.add_argument("--n_hid", default=256, type=int)
    parser.add_argument("--n_layers", default=2, type=int)
    parser.add_argument("--n_train_steps", default=10000, type=int)
    parser.add_argument("--num_init_points", default=64, type=int)
    parser.add_argument("--num_val_points", default=128, type=int)
    parser.add_argument("--num_iter", default=10, type=int)
    parser.add_argument("--num_val_iters", default=10, type=int)
    parser.add_argument("--reward_topk", default=5, type=int)
    parser.add_argument("--inf_batch_size", default=32, type=int)
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
    tfc = lambda x: torch.FloatTensor(x)
    tl = lambda x: torch.LongTensor(x).to(dev)

    TEST_FUNC = (lambda x: ((np.cos(x * 50 - 2) * np.sin(10 * x + 2) + 1) * norm.pdf(x * 5, loc=2, scale=2)) + 0.01)
    LOGINF = torch.tensor(1000).to(dev)
    main(args)
