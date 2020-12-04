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
from ray.rllib.agents.dqn.dqn_torch_policy import build_q_losses, build_q_model_and_distribution, \
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



