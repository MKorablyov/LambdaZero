# from torch.multiprocessing import Pool, Process, set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass
import copy
import os
import time
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

from lightrl.a2c_ppo_acktr import algo, utils
from lightrl.a2c_ppo_acktr.algo import gail

from lightrl.utils.utils import parse_opts, add_to_cfg, flatten_cfg, update_cfg
from lightrl.env.vec_env import get_envs, pre_process_obss
from lightrl import env
from lightrl.models import get_model
from lightrl.policy.policy_base import Policy
from lightrl.utils.storage import RolloutStorage
from lightrl.evaluation import EvaluateBase

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



def preprocess_args(args: Namespace):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
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
        assert hasattr(args.env_cfg.env_args, "env_seed"), f"Must have {env_seed}"

    # Generate eval env cfg from env_cfg and eval_env_cfg
    eval_cfg = copy.deepcopy(args.env_cfg)
    update_cfg(eval_cfg, args.eval_env_cfg) # inplace update with new cfg
    args.eval_env_cfg = eval_cfg
    if eval_seeds is not None:
        args.eval_env_cfg.env_args.env_seed = eval_seeds
    # ==============================================================================================

    return args


def set_seed(seed: int, use_cuda: bool, cuda_deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if use_cuda and torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def setup_loggers(args: Namespace):
    # Load loggers and setup results path
    use_wandb = args.use_wandb

    if use_wandb:
        experiment_name = f"{args.full_title}_{args.run_id}"
        with open(f"{summaries_dir}/wandb_key") as f:
            api_key = f.readlines()[0].strip()

        os.environ['WANDB_API_KEY'] = api_key

        wandb.init(project="rlbo4", name=experiment_name)
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


def run(args):
    args = preprocess_args(args)

    set_seed(args.seed, args.cuda, args.cuda_deterministic)

    setup_loggers(args)
    do_plot = args.use_wandb

    # torch.set_num_threads(1)  # TODO why?
    device = args.device

    print("Load training envs ...")
    envs, chunk_size = get_envs(args.env_cfg, args.env_cfg.procs)

    def process_obss(x):
        return pre_process_obss(x, device=device)


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

    episode_rewards = deque(maxlen=100)

    # DEBUG - -----------
    seed_to_smiles = dict()
    for k, v in evaluation._eval_data.items():
        seed_to_smiles[v["env_seed"][0]] = k
    actor_critic.eval_data = evaluation._eval_data
    actor_critic.seed_to_smiles = seed_to_smiles
    # DEBUG - -----------

    print("Start training ...")
    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            """
            # Sample actions
            with torch.no_grad():
                res_m = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                value, action = res_m.value, res_m.action
                action_log_prob, recurrent_hidden_states = res_m.action_log_probs, res_m.rnn_hxs

            send_act = action.flatten().cpu().numpy().tolist()
            """
            send_act = [envs.action_space.sample() for _ in range(len(obs))]
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(send_act)

            for iobs, info in enumerate(infos):
                if done[iobs]:
                    episode_rewards.append(reward[iobs])

                # if 'episode' in info.keys():
                #     episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            reward = torch.FloatTensor(reward).unsqueeze(1)

            # p_obs = pre_process_obss(obs, device=device)
        """
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
        """

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

        """
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        """

        value_loss, action_loss, dist_entropy = 0, 0, 0
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

        # wandb.log({'timesteps_total': (j + 1) * args.num_processes * args.num_steps})
        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps,
                            int(total_num_steps / (end - start)),
                            len(episode_rewards), np.mean(episode_rewards),
                            np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss,
                            action_loss))


        plot_vals = dict({
            "total_num_steps": total_num_steps,
            "train_iter": j,
            "reward_mean": np.mean(episode_rewards),
        })

        if args.eval_interval != 0 and j % args.eval_interval == 0:
            scores = evaluation.evaluate(j ,total_num_steps)
            plot_vals.update(scores.mean().to_dict())

        if do_plot:
            wandb.log(plot_vals)

        #     obs_rms = utils.get_vec_normalize(envs).obs_rms
        #     evaluate(actor_critic, obs_rms, args.env_name, args.seed,
        #              args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    import sys as _sys

    args = _sys.argv[1:]
    args += ["--session-id", "0", "--results-path", f"{summaries_dir}/lightrl"]
    print(parse_opts(args))
    exit()
    run(parse_opts())
