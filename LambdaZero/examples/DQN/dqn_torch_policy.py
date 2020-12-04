from gym.spaces import Discrete

import ray
from ray.rllib.agents.dqn.dqn_tf_policy import postprocess_nstep_and_prio, \
    PRIO_WEIGHTS, Q_SCOPE, Q_TARGET_SCOPE
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.agents.dqn.simple_q_torch_policy import TargetNetworkMixin
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy import LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import huber_loss, reduce_mean_ignore_inf, \
    softmax_cross_entropy_with_logits

from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG, validate_config, execution_plan # Leo: edit
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_model_and_distribution, \
    build_q_stats, postprocess_nstep_and_prio, adam_optimizer, grad_process_and_td_error_fn, \
    extra_action_out_fn, \
    setup_early_mixins, after_init, TargetNetworkMixin, ComputeTDErrorMixin, LearningRateSchedule
from ray.rllib.agents.trainer_template import build_trainer

import random
import numpy.ma as ma
import numpy as np

torch, nn = try_import_torch()

eps = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_action_sampler(policy, model, obs_batch, explore, timestep, is_training=False):
    # We are ignoring explore in this case
    q_vals, _, action_mask = compute_q_values(policy, model, obs_batch, explore, is_training)
    policy.q_values = q_vals
    # Ensure that you don't sample the actions that are illegal (i.e. action_mask... )
    action_mask = (1. - action_mask).squeeze(-1).cpu().numpy().astype(np.bool)
    masked_q_vals = ma.masked_array(q_vals.cpu().numpy(), action_mask, fill_value=1e-20)

#    import pdb; pdb.set_trace()
    
    # Epsilon-Greedy
    if random.random() < eps:
        try:
            allowed_actions = (1 - action_mask).astype(np.bool)
            actions = np.array([np.random.choice(np.arange(q_vals.shape[1])[allowed_actions[i]]) for i in range(allowed_actions.shape[0])])
        except:
            import pdb; pdb.set_trace()
    else:
        actions = masked_q_vals.argmax(axis=1) 
    return torch.tensor(actions).to(device), None # returns actions, logp


def compute_q_values(policy, model, obs, explore, is_training=False):
    config = policy.config

    model_out, state = model({ 
        SampleBatch.CUR_OBS: obs, 
        "is_training": is_training,
    }, [], None) # Where's the action? Q(s, a), but I don't see the action...

    (action_scores, logits,
     probs_or_logits) = model.get_q_value_distributions(model_out) # Not sure how logits or (probs or logits) are anything but 1

    if config["dueling"]: # TODO
        state_score = model.get_state_value(model_out)
        if policy.config["num_atoms"] > 1:
            support_logits_per_action_mean = torch.mean(
                support_logits_per_action, dim=1)
            support_logits_per_action_centered = (
                support_logits_per_action - torch.unsqueeze(
                    support_logits_per_action_mean, dim=1))
            support_logits_per_action = torch.unsqueeze(
                state_score, dim=1) + support_logits_per_action_centered
            support_prob_per_action = nn.functional.softmax(
                support_logits_per_action)
            value = torch.sum(z * support_prob_per_action, dim=-1)
            logits = support_logits_per_action
            probs_or_logits = support_prob_per_action
        else:
            advantages_mean = reduce_mean_ignore_inf(action_scores, 1)
            advantages_centered = action_scores - torch.unsqueeze(
                advantages_mean, 1)
            value = state_score + advantages_centered
    else:
        value = action_scores # Literally just this

    return value, logits, probs_or_logits

def build_q_losses(policy, model, _, train_batch):
    config = policy.config
    # Q-network evaluation.
    q_t, q_logits_t, q_probs_t = compute_q_values(
        policy,
        policy.q_model,
        train_batch[SampleBatch.CUR_OBS],
        explore=False,
        is_training=True)

    # Target Q-network evaluation.
    q_tp1, q_logits_tp1, q_probs_tp1 = compute_q_values(
        policy,
        policy.target_q_model,
        train_batch[SampleBatch.NEXT_OBS],
        explore=False,
        is_training=True)

    # Q scores for actions which we know were selected in the given state.
    one_hot_selection = F.one_hot(
        train_batch[SampleBatch.ACTIONS], policy.action_space.n)
    q_t_selected = torch.sum(
        torch.where(q_t > -float("inf"), q_t, torch.tensor(0.0).to(device)) *
        one_hot_selection, 1)
    q_logits_t_selected = torch.sum(
        q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1)

    # compute estimate of best possible value starting from state at t + 1
    if config["double_q"]:
        q_tp1_using_online_net, q_logits_tp1_using_online_net, \
            q_dist_tp1_using_online_net = compute_q_values(
                policy,
                policy.q_model,
                train_batch[SampleBatch.NEXT_OBS],
                explore=False,
                is_training=True)
        q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
        q_tp1_best_one_hot_selection = F.one_hot(
            q_tp1_best_using_online_net, policy.action_space.n)
        q_tp1_best = torch.sum(
            torch.where(q_tp1 > -float("inf"), q_tp1, torch.tensor(0.0).to(device)) *
            q_tp1_best_one_hot_selection, 1)
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1)
    else:
        q_tp1_best_one_hot_selection = F.one_hot(
            torch.argmax(q_tp1, 1), policy.action_space.n)
        q_tp1_best = torch.sum(
            torch.where(q_tp1 > -float("inf"), q_tp1, torch.tensor(0.0).to(device)) *
            q_tp1_best_one_hot_selection, 1)
        q_probs_tp1_best = torch.sum(
            q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1)

    policy.q_loss = QLoss(
        q_t_selected, q_logits_t_selected, q_tp1_best, q_probs_tp1_best,
        train_batch[PRIO_WEIGHTS], train_batch[SampleBatch.REWARDS],
        train_batch[SampleBatch.DONES].float(), config["gamma"],
        config["n_step"], config["num_atoms"],
        config["v_min"], config["v_max"])

    return policy.q_loss.loss

def get_policy_class(config):
    # Overriding the original
    return CustomDQNTorchPolicy


CustomDQNTorchPolicy = build_torch_policy(
    name="CustomDQNTorchPolicy",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.agents.dqn.dqn.DEFAULT_CONFIG,
    make_model_and_action_dist=build_q_model_and_distribution,
    action_distribution_fn=None,
    action_sampler_fn=get_action_sampler,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.q_loss.td_error},
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    after_init=after_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ])

GenericOffPolicyTrainer = build_trainer(
    name="GenericOffPolicyAlgorithm",
    default_policy=None,
    get_policy_class=get_policy_class,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    execution_plan=execution_plan)


CustomDQNTrainer = GenericOffPolicyTrainer.with_updates(
    name="CustomDQNTrainer", default_policy=CustomDQNTorchPolicy, default_config=DEFAULT_CONFIG)



