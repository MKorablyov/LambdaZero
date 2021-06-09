from asymmetric_self_play_agents import Alice, Bob, NewAlice
from toy_1d import BinaryTreeEnv
import argparse

import os
import logging
import logging.handlers
import torch

from torch.utils.tensorboard import SummaryWriter

__all__ = ['init_logger']


def init_logger(log_dir, log_name):
    os.makedirs(log_dir, exist_ok=True)
    log_level = logging.DEBUG
    log_format = '%(filename)s | [line:%(lineno)d] | %(asctime)s | %(message)s'

    logger = logging.getLogger(log_dir)
    logger.setLevel(log_level)
    path = os.path.join(log_dir, log_name)

    # file handler (log file)
    log_handler = logging.handlers.RotatingFileHandler(filename=path)
    log_handler.setLevel(log_level)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)

    # stream handler (default sys.stderr)
    log_handler = logging.StreamHandler()
    log_handler.setLevel(log_level)
    log_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(log_handler)

    return logger


parser = argparse.ArgumentParser()

# ====== Environment arguments
parser.add_argument('--horizon', type=int, default=10,
                    help='horizon for tree env')

# ====== Alice Agent Arguments ======
parser.add_argument('--alice_type', default='ppo',
                    help='type of Alice agent to use')

# ====== PPO Arguments ======
parser.add_argument('--clip_param', type=float, default=0.2,
                    help='clip parameter for PPO')
parser.add_argument('--p_lr', type=float, default=1e-4,
                    help='policy learning rate for PPO')
parser.add_argument('--v_lr', type=float, default=1e-4,
                    help='value learning rate for PPO')
parser.add_argument('--num_trajs', type=int, default=5,
                    help='number of trajectories for Alice agent to rollout')
parser.add_argument('--num_training_steps', type=int, default=100,
                    help='number of iterations to run the whole loop')

# ====== BC arguments (Bob)
parser.add_argument('--bc_lr', type=float, default=1e-4,
                    help='BC learning rate for Bob')
parser.add_argument('--bc_epochs', type=int, default=10,
                    help='number of epochs for BC to learn')

# ====== Logging Arguments ======
parser.add_argument('--log_dir', default='./asymmetric_self_play_logs',
                    help='logging directory for experiment logs')
parser.add_argument('--log_name', default='main.log', help='name of log file')
parser.add_argument('--tensorboard_logdir', default='./asymmetric_self_play_tensorboard_logs',
                    help='directory for experiment tensorboard logs')

# ====== Saving Arguments ======
parser.add_argument(
    '--alice_save_path', default='./asp_models/alice.pt', help='alice agent save path')
parser.add_argument(
    '--bob_save_path', default='./asp_models/bob.pt', help='bob agent save path')

args = parser.parse_args()

# ====== Environment setup ======
env = BinaryTreeEnv(args.horizon)

# ====== Agents ======
if args.alice_type == 'ppo':
    alice = Alice(env, p_lr=args.p_lr, v_lr=args.v_lr)
else:
    alice = NewAlice(env, p_lr=args.p_lr)
alice.train()
bob = Bob(env, lr=args.bc_lr)
bob.train()

# ====== Rollout collection function ======


def collect_rollout_data(args):
    # initialize Bob reached to be True to start
    bob_reached = True

    # Alice trajectories
    alice_trajs = []

    # Bob trajectories
    bob_trajs = []

    # BC trajectories
    bc_trajs = []

    for _ in range(args.num_trajs):
        a_traj, goal = alice.collect_trajectory()

        if bob_reached:
            b_traj, bob_reached = bob.collect_trajectory(goal)
            bob_trajs.append((b_traj, bob_reached))

        alice_reward = 5 if not bob_reached else 0

        # augment reward for Alice trajectory, first by multiplying by alice reward, then dividing by prob that Bob got to that goal
        ft = a_traj[-1]
        # tensor with just the probability that Bob got there
        bob_prob = bob.get_traj_probs([(a_traj, goal)])
        new_reward = ft[2] * alice_reward / bob_prob[0]
        # reward is r_final * alice_reward, which means it's bad if r_final is negative (regardless of what alice_reward is) and good if r_final is > 0
        new_transition = (ft[0], ft[1], new_reward, ft[3], ft[4])
        a_traj[-1] = new_transition

        alice_trajs.append((a_traj, goal))

        # now if Bob didn't reach the goal, then we can use BC and just imitate the failed Alice trajectories
        if not bob_reached:
            bc_traj = []
            for t in a_traj:
                new_transition = (t[0], t[1], t[2], t[3], t[4], goal)
                bc_traj.append(new_transition)

            bc_trajs.append(bc_traj)

    return alice_trajs, bob_trajs, bc_trajs


def main(args):
    # ====== Initialize logger and tensorboard writer ======
    logger = init_logger(args.log_dir, args.log_name)
    writer = SummaryWriter(args.tensorboard_logdir)

    for i in range(args.num_training_steps):
        a_trajs, _, _ = collect_rollout_data(args)

        # update both agents, Bob first

        # ====== Behavioral cloning training (Bob update) ======
        for bc_step in range(args.bc_epochs):
            bc_loss = bob.update(a_trajs)
            writer.add_scalar('behavioral_cloning_loss', bc_loss,
                              args.num_training_steps * i + bc_step)  # we expect to see this oscillating between high and low losses

        # ====== Alice update ======
        if args.alice_type == 'ppo':
            ppo_objective, value_loss = alice.update_ppo(
                a_trajs, args.clip_param)
            logger.info(f'PPO objective after epoch {i+1}: {ppo_objective}')
            logger.info(
                f'Value loss (MSE loss) after epoch {i+1}: {value_loss}')
            writer.add_scalar('ppo_objective', ppo_objective, i + 1)
            writer.add_scalar('value_mse_loss', value_loss, i + 1)
        else:
            # now that Bob has successfully imitated, it should assign high probability to all transitions from Alice so far
            bob_probs = bob.get_traj_probs(a_trajs)
            alice_objective = alice.update(a_trajs, bob_probs)
            logger.info(f'Alice objective after epoch {i+1}: {alice_objective}')
            writer.add_scalar('alice_new_objective', alice_objective, i + 1)

        logger.info(
            f'Behavioral cloning loss (Bob loss) after epoch {i+1}: {bc_loss}')

    # saving
    if args.alice_type == 'ppo':
        alice_stuff = {
            'policy': alice.policy,
            'value': alice.value,
            'policy_optimizer': alice.policy_optimizer,
            'value_optimizer': alice.value_optimizer
        }
    else:
        alice_stuff = {
            'policy': alice.policy,
            'policy_optimizer': alice.policy_optimizer
        }

    bob_stuff = {
        'policy': bob.policy,
        'optimizer': bob.optimizer
    }

    torch.save(alice_stuff, args.alice_save_path)
    torch.save(bob_stuff, args.bob_save_path)


if __name__ == '__main__':
    main(args)
