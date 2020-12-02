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
    extra_action_out_fn, compute_q_values, \
    setup_early_mixins, after_init, TargetNetworkMixin, ComputeTDErrorMixin, LearningRateSchedule
from ray.rllib.agents.trainer_template import build_trainer


def get_action_sampler(policy, model, obs_batch, explore, timestep, is_training=False):
    q_vals = compute_q_values(policy, model, obs_batch, explore, is_training)
    q_vals = q_vals[0] if isinstance(q_vals, tuple) else q_vals
    policy.q_values = q_vals
    # Ensure that you don't sample the actions that are illegal (i.e. action_mask... )
    actions = q_vals.argmax(dim=1)
    return actions, None # returns actions, logp



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



