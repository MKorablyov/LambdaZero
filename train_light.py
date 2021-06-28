import cv2
import collections

from torch.multiprocessing import Pool, Process, set_start_method
# set_start_method('spawn')
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import copy
import os
import time
import collections
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import Namespace
import wandb
import csv
from typing import List

from lightrl.a2c_ppo_acktr import algo, utils
from lightrl.a2c_ppo_acktr.algo import gail

from lightrl.utils.utils import parse_opts, add_to_cfg, flatten_cfg, update_cfg
from lightrl.env.vec_env import get_envs, pre_process_obss
from lightrl import env
from lightrl.models import get_model
from lightrl.policy.policy_base import Policy
from lightrl.utils.storage import RolloutStorage
from lightrl.evaluation import EvaluateBase
from lightrl.utils.utils import set_seed
from lightrl.utils.utils import LogTopStats, LogStatsTrain, SummaryStats
from lightrl.env.oracle import InterogateOracle
from multiprocessing import Process, Pipe, Queue

from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


LOG_HEADER = {
    "update": None,
    "timesteps": None,
    "median": None,
    "reward": None,
    "min": None,
    "max": None,
    "dist_entropy": None,
    "value_loss": None,
    "action_loss": None,
}

MAIN_CFG_ARGS = ["model", "env_cfg"]


class TimeStats:
    def __init__(self):
        self._buffer = []
        self._max = 1000
        self._st_ts = time.time()

    def start(self):
        self._st_ts = time.time()

    def end(self):
        fin = time.time() - self._st_ts
        self._buffer.append(fin)
        if len(self._buffer) > self._max:
            self._buffer.pop(0)
        return np.mean(self._buffer)

    def stats(self):
        return np.mean(self._buffer)


def preprocess_args(args: Namespace):
    args.cuda = args.main.use_gpu and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.main.seed == 0:
        args.main.seed = args.run_id + 1

    print(f"SEED: {args.main.seed}")

    add_to_cfg(args, None, "device", args.device)
    add_to_cfg(args, None, "seed", args.main.seed)
    args.seed = args.main.seed

    args.num_processes = args.env_cfg.procs

    # ==============================================================================================
    # Get training and test seeds
    eval_seeds = None
    if hasattr(args.env_cfg.env_args, "seed_cnt"):
        from LambdaZero.contrib.config_eval_env import ALL_SEEDS, SEED_FILTER
        seed_cnt = args.env_cfg.env_args.seed_cnt
        assert seed_cnt < len(ALL_SEEDS), "Do not have enough pre-stored seeds"
        train_seeds = ALL_SEEDS[:seed_cnt].tolist()
        eval_seeds = train_seeds  # ALL_SEEDS[:seed_cnt][SEED_FILTER[:seed_cnt]].tolist() # TODO eval only worth eval
        args.env_cfg.env_args.env_seed = train_seeds
    else:
        assert hasattr(args.env_cfg.env_args, "env_seed"), f"Must have env_seed"

    # Generate eval env cfg from env_cfg and eval_env_cfg
    eval_cfg = copy.deepcopy(args.env_cfg)
    update_cfg(eval_cfg, args.eval_env_cfg) # inplace update with new cfg
    args.eval_env_cfg = eval_cfg
    if eval_seeds is not None:
        args.eval_env_cfg.env_args.env_seed = eval_seeds
    # ==============================================================================================

    return args


def setup_loggers(args: Namespace):
    # Load loggers and setup results path
    use_wandb = args.main.plot

    if use_wandb:
        experiment_name = f"{args.full_title}_{args.run_id}"
        with open(f"{summaries_dir}/wandb_key") as f:
            api_key = f.readlines()[0].strip()

        os.environ['WANDB_API_KEY'] = api_key

        wandb.init(project="an_rlbo", name=experiment_name)
        wandb.config.update(dict(flatten_cfg(args)))

    out_dir = args.out_dir

    if args.log_monitor:
        # OPENAI Monitor wrapper
        os.environ['OPENAI_LOGDIR'] = args.out_dir

        log_dir = f"{out_dir}/monitor"
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        args.monitor_dir = log_dir

    flog = open(out_dir + "/logs.csv", 'w')
    log_writer = csv.DictWriter(flog, LOG_HEADER.keys())
    log_writer.writeheader()


def log_stats_remote(conn_recv, conn_send, log_topk):
    log_stats = LogTopStats(topk=log_topk)

    while True:
        cmd, recv = conn_recv.get()
        if cmd == 0:
            log_stats.collect(*recv)
        elif cmd == 1:
            stats = log_stats.log()
            stats["total_num_steps"] = recv
            conn_send.put(stats)


def run(args):
    print("SLURM_ARRAY_TASK_ID", os.environ.get('SLURM_ARRAY_TASK_ID', None))
    print("SLURM_ARRAY_JOB_ID", os.environ.get('SLURM_ARRAY_JOB_ID', None))

    args = preprocess_args(args)
    device = args.device
    do_plot = args.main.plot

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    # ==============================================================================================
    # Setup loggers

    setup_loggers(args)  # Wandb plot mostly and maybe writers to csv

    # Log train rl stuff
    log_train = LogStatsTrain(r_buffer_size=args.main.stats_window_size)

    # Get remote calculation of logs for topk stats (because we also calculate dockscore)
    log_stats_remote_send = Queue()
    log_stats_remote_recv = Queue()
    log_proc_stats = Process(
        target=log_stats_remote,
        args=(log_stats_remote_send, log_stats_remote_recv, args.main.log_topk)
    )
    log_proc_stats.start()

    # This logs statistics for MIN and MAX values for everything plotted (summary for wandb)
    log_summary = SummaryStats(log_wandb=do_plot)

    # ==============================================================================================
    # -- Reward

    from lightrl.env.scores import ParallelSynth, QEDEstimator
    synth_net = ParallelSynth(use_cuda=args.main.use_gpu)
    synth_net.to(device)
    synth_net.share_memory()
    args.env_cfg.env_args.synth_net = synth_net
    args.eval_env_cfg.env_args.synth_net = synth_net

    if args.proxy_dock is not None:
        if args.proxy_dock != "default":
            from lightrl.env.scores import load_docknet
            proxy_net = load_docknet(args.proxy_dock, device=device)
        else:
            from lightrl.env.scores import LZProxyDockNetRun
            proxy_net = LZProxyDockNetRun()
            proxy_net.to(device)

        proxy_net.share_memory()
        args.env_cfg.env_args.proxy_net = proxy_net
        args.eval_env_cfg.env_args.proxy_net = proxy_net


    # ==============================================================================================

    print("Load training envs ...")
    envs, chunk_size = get_envs(args.env_cfg, args.env_cfg.procs)

    def process_obss(x):
        return pre_process_obss(x, device=device)

    from lightrl.reg_models import get_actor_model
    from lightrl import reg_models

    if args.model.name in reg_models.MODELS:
        base_model = get_actor_model(args.model)
    else:
        base_model = get_model(args.model, envs.observation_space.shape, envs.action_space)

    actor_critic = Policy(args.policy, envs.observation_space.shape, envs.action_space, base_model)
    actor_critic.to(device)

    print("Load Evaluation protocol and envs ...")
    evaluation = EvaluateBase(actor_critic, process_obss, args.eval_env_cfg, "", device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              process_obss)

    obs = envs.reset()

    rollouts._obs[0] = obs
    rollouts.to(device)

    # DEBUG - -----------
    seed_to_smiles = dict()
    for k, v in evaluation._eval_data.items():
        seed_to_smiles[v["env_seed"][0]] = k
    actor_critic.eval_data = evaluation._eval_data
    actor_critic.seed_to_smiles = seed_to_smiles
    out_dir = args.out_dir
    data_out_dir = f"{out_dir}/train_data"
    os.mkdir(data_out_dir)

    ts_stats = Namespace(
        fwd=TimeStats(),
        act=TimeStats(),
        update=TimeStats(),
        insert=TimeStats(),
        nextv=TimeStats(),
        compute_r=TimeStats(),
        after_update=TimeStats(),
    )
    # DEBUG - ----------
    # ==============================================================================================

    print("Start training ...")
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    max_data_out = f"{out_dir}/max_data"
    os.mkdir(max_data_out)

    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            ts_stats.fwd.start()
            with torch.no_grad():
                res_m = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                value, action = res_m.value, res_m.action
                action_log_prob, recurrent_hidden_states = res_m.action_log_probs, res_m.rnn_hxs
            ts_stats.fwd.end()

            send_act = action.flatten().cpu().numpy().tolist()

            # Obser reward and next obs
            ts_stats.act.start()
            obs, reward, done, infos = envs.step(send_act)
            ts_stats.act.end()

            new_mol_infos = log_train.collect_new_batch(obs, reward, done, infos)

            # Send training step new mol infos for topK log statistics
            if len(new_mol_infos) > 0:
                log_stats_remote_send.put((0, (new_mol_infos,)))

            ts_stats.insert.start()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0]
                                           for info in infos])
            reward = torch.FloatTensor(reward).unsqueeze(1)

            # p_obs = pre_process_obss(obs, device=device)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            ts_stats.insert.end()

        ts_stats.nextv.start()
        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        ts_stats.nextv.end()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        ts_stats.compute_r.start()
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)
        ts_stats.compute_r.end()

        ts_stats.update.start()
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        ts_stats.update.end()

        ts_stats.after_update.start()
        rollouts.after_update()
        ts_stats.after_update.end()

        # # save for every interval-th episode or for the last epoch
        # if (j % args.save_interval == 0
        #     or j == num_updates - 1) and args.save_dir != "":
        #     save_path = os.path.join(args.save_dir, args.algo)
        #     try:
        #         os.makedirs(save_path)
        #     except OSError:
        #         pass
        #
        #     torch.save([
        #         actor_critic,
        #         getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
        #     ], os.path.join(save_path, args.env_name + ".pt"))
        #

        # Get training log
        plot_vals, new_mol_infos = log_train.log()

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if (j + 1) % args.main.log_topk_interval == 0:
            log_stats_remote_send.put((1, total_num_steps))

        while not log_stats_remote_recv.empty():
            log_stats = log_stats_remote_recv.get()
            print(log_stats)
            log_summary.update(log_stats)
            if do_plot:
                wandb.log(log_stats)

        if j % 20 == 0:
            for k, v in ts_stats.__dict__.items():
                print(f"{k}: {v.stats():.6f}")

        end = time.time()
        plot_vals.update({
            "fps": int(total_num_steps / (end - start)),
            "dist_entropy": dist_entropy,
            "value_loss": value_loss,
            "action_loss": action_loss,
            "total_num_steps": total_num_steps,
            "train_iter": j,
        })

        if j % args.log_interval == 0:
            print(plot_vals)

        if args.eval_interval != 0 and j % args.eval_interval == 0:
            scores = evaluation.evaluate(j ,total_num_steps)
            plot_vals.update(scores.mean().to_dict())

        if do_plot:
            wandb.log(plot_vals)


if __name__ == "__main__":
    from lightrl.utils.utils import parse_opts
    run(parse_opts())
