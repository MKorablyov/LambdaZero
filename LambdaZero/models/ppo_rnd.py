import logging
import numpy as np
import scipy

import ray
from ray.rllib.agents.a3c.a3c_torch_policy import apply_grad_clipping
from ray.rllib.agents.ppo.ppo_tf_policy import setup_config
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.torch_policy_template import build_torch_policy
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils.torch_ops import sequence_mask
from ray.rllib.agents.ppo.ppo import DEFAULT_CONFIG, execution_plan, validate_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models.modelv2 import restore_original_dimensions

torch, nn = try_import_torch()

logger = logging.getLogger(__name__)


class PPORNDLoss:
    def __init__(self,
                 dist_class,
                 model,
                 value_targets,
                 value_int_targets,
                 advantages_ext,
                 advantages_int,
                 actions,
                 prev_logits,
                 prev_actions_logp,
                 vf_preds,
                 vf_int_preds,
                 curr_action_dist,
                 value_fn,
                 int_value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 rnd_weight=1.0,
                 use_gae=True):
        """Constructs the loss for Proximal Policy Objective.
        Arguments:
            dist_class: action distribution class for logits.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            prev_logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            prev_actions_logp (Placeholder): Placeholder for prob output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        if valid_mask is not None:
            num_valid = torch.sum(valid_mask)

            def reduce_mean_valid(t):
                return torch.sum(t * valid_mask) / num_valid

        else:

            def reduce_mean_valid(t):
                return torch.mean(t)

        advantages = 0.75 * advantages_ext + 0.25 * advantages_int

        prev_dist = dist_class(prev_logits, model)
        # Make loss functions.
        logp_ratio = torch.exp(
            curr_action_dist.logp(actions) - prev_actions_logp)
        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        curr_entropy = curr_action_dist.entropy()
        self.mean_entropy = reduce_mean_valid(curr_entropy)

        surrogate_loss = torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1 - clip_param,
                                     1 + clip_param))
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        if use_gae:
            vf_loss1 = torch.pow(value_fn - value_targets, 2.0)
            vf_clipped = vf_preds + torch.clamp(value_fn - vf_preds,
                                                -vf_clip_param, vf_clip_param)
            vf_loss2 = torch.pow(vf_clipped - value_targets, 2.0)
            vf_loss = torch.max(vf_loss1, vf_loss2)
            
            # RND
            vf_int_loss1 = torch.pow(int_value_fn - value_int_targets, 2.0)
            vf_int_clipped = vf_preds + torch.clamp(int_value_fn - vf_int_preds,
                                                -vf_clip_param, vf_clip_param)
            vf_int_loss2 = torch.pow(vf_int_clipped - value_int_targets, 2.0)
            vf_int_loss = torch.max(vf_int_loss1, vf_int_loss2)
            
            self.mean_vf_loss = reduce_mean_valid(vf_loss)
            self.mean_vf_int_loss = reduce_mean_valid(vf_int_loss)

            loss = reduce_mean_valid(
                -surrogate_loss + cur_kl_coeff * action_kl +
                vf_loss_coeff * (vf_loss + vf_int_loss) - entropy_coeff * curr_entropy)
        else:
            self.mean_vf_loss = 0.0
            loss = reduce_mean_valid(-surrogate_loss +
                                     cur_kl_coeff * action_kl -
                                     entropy_coeff * curr_entropy)
        self.loss = loss

def discount(x: np.ndarray, gamma: float):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def lfilter(input_tensor, a_coeffs, b_coeffs):
    # pack batch
    shape = input_tensor.size()
    input_tensor = input_tensor.view(-1, shape[-1])

    device = input_tensor.device
    dtype = input_tensor.dtype
    n_channel, n_sample = input_tensor.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_input = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_input[:, (n_order - 1):] = input_tensor
    padded_output = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_input, window_idxs))

    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output[:, i_sample:(i_sample + n_order)]
        o0.sub_(torch.mv(windowed_output_signal, a_coeffs_flipped))
        o0.div_(a_coeffs[0])

        padded_output[:, i_sample + n_order - 1] = o0

    output = torch.clamp(padded_output[:, (n_order - 1):], min=-1., max=1.)

    # unpack batch
    output = output.view(shape[:-1] + output.shape[-1:])

    return output


def ppo_rnd_surrogate_loss(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    train_batch["int_reward"] = model.compute_intrinsic_rewards(train_batch)

    if policy.config["use_gae"]:
        vpred_t = torch.cat([train_batch["vf_int_pred"], torch.tensor([0.], device=train_batch["vf_int_pred"].device)])
        delta_t = (
            train_batch["int_reward"] + policy.config['gamma'] * vpred_t[1:] - vpred_t[:-1])
        a_coeff = torch.tensor([1.0, 0.0]).to(delta_t[:-1].device)
        b_coeff = torch.tensor([1.0, - policy.config['gamma'] * policy.config['lambda']]).to(delta_t[:-1].device)
        train_batch["int_adv"] = lfilter(delta_t, a_coeff, b_coeff)
        train_batch["vf_int_target"] = (train_batch["int_adv"] + train_batch["vf_int_pred"])


    else:
        int_rewards_plus_v = np.concatenate(
            [rollout["int_reward"],
             np.array([0.])])
        int_discounted_returns = discount(int_rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)

        train_batch["int_adv"] = int_discounted_returns - rollout["vf_int_pred"]
        train_batch["vf_int_target"] = int_discounted_returns


    # train_batch["int_adv"] = train_batch["int_adv"].clone().detach().cpu().numpy()

    mask = None
    if state:
        max_seq_len = torch.max(train_batch["seq_lens"])
        mask = sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = torch.reshape(mask, [-1])

    policy.loss_obj = PPORNDLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch["vf_int_target"],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch["int_adv"],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        train_batch["vf_int_pred"],
        action_dist,
        model.value_function(),
        model.int_value_function(),
        policy.kl_coeff,
        mask,
        entropy_coeff=policy.entropy_coeff,
        clip_param=policy.config["clip_param"],
        vf_clip_param=policy.config["vf_clip_param"],
        vf_loss_coeff=policy.config["vf_loss_coeff"],
        rnd_weight=model.rnd_weight,
        use_gae=policy.config["use_gae"],
    )

    return policy.loss_obj.loss


def kl_and_loss_stats(policy, train_batch):
    return {
        "cur_kl_coeff": policy.kl_coeff,
        "cur_lr": policy.cur_lr,
        "total_loss": policy.loss_obj.loss,
        "policy_loss": policy.loss_obj.mean_policy_loss,
        "vf_loss": policy.loss_obj.mean_vf_loss,
        "vf_int_loss": policy.loss_obj.mean_vf_int_loss,
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS],
            policy.model.value_function(),
            framework="torch"),
        "kl": policy.loss_obj.mean_kl,
        "entropy": policy.loss_obj.mean_entropy,
        "entropy_coeff": policy.entropy_coeff,
    }

def vf_preds_fetches(policy, input_dict, state_batches, model, action_dist):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.model.value_function(),
        "vf_int_pred": policy.model.int_value_function(),
        # "int_reward": policy.model.compute_intrinsic_rewards(input_dict)
    }


class KLCoeffMixin:
    def __init__(self, config):
        # KL Coefficient.
        self.kl_coeff = config["kl_coeff"]
        self.kl_target = config["kl_target"]

    def update_kl(self, sampled_kl):
        if sampled_kl > 2.0 * self.kl_target:
            self.kl_coeff *= 1.5
        elif sampled_kl < 0.5 * self.kl_target:
            self.kl_coeff *= 0.5
        return self.kl_coeff


class ValueNetworkMixin:
    def __init__(self, obs_space, action_space, config):
        if config["use_gae"]:

            def value(ob, prev_action, prev_reward, *state):
                model_out, _ = self.model({
                    SampleBatch.CUR_OBS: self._convert_to_tensor([ob]),
                    SampleBatch.PREV_ACTIONS: self._convert_to_tensor(
                        [prev_action]),
                    SampleBatch.PREV_REWARDS: self._convert_to_tensor(
                        [prev_reward]),
                    "is_training": False,
                }, [self._convert_to_tensor(s) for s in state],
                                          self._convert_to_tensor([1]))
                return self.model.value_function()[0]

        else:

            def value(ob, prev_action, prev_reward, *state):
                return 0.0

        self._value = value


def setup_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"],
                                  config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])


def compute_advantages(rollout: SampleBatch,
                       last_r: float,
                       gamma: float = 0.9,
                       lambda_: float = 1.0,
                       use_gae: bool = True,
                       use_critic: bool = True):
    """
    Given a rollout, compute its value targets and the advantage.
    Args:
        rollout (SampleBatch): SampleBatch of a single trajectory
        last_r (float): Value estimation for last observation
        gamma (float): Discount factor.
        lambda_ (float): Parameter for GAE
        use_gae (bool): Using Generalized Advantage Estimation
        use_critic (bool): Whether to use critic (value estimates). Setting
                           this to False will use 0 as baseline.
    Returns:
        SampleBatch (SampleBatch): Object with experience from rollout and
            processed rewards.
    """
    traj = {}
    trajsize = len(rollout[SampleBatch.ACTIONS])
    for key in rollout:
        traj[key] = np.stack(rollout[key])

    assert SampleBatch.VF_PREDS in rollout or not use_critic, \
        "use_critic=True but values not found"
    assert use_critic or not use_gae, \
        "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate(
            [rollout[SampleBatch.VF_PREDS],
             np.array([last_r])])
        delta_t = (
            traj[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1])
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        traj[Postprocessing.ADVANTAGES] = discount(delta_t, gamma * lambda_)
        traj[Postprocessing.VALUE_TARGETS] = (
            traj[Postprocessing.ADVANTAGES] +
            traj[SampleBatch.VF_PREDS]).copy().astype(np.float32)

        # RND
        # vpred_t = np.concatenate(
        #     [rollout["vf_int_pred"],
        #      np.array([0.])])
        # delta_t = (
        #     traj["int_reward"] + gamma * vpred_t[1:] - vpred_t[:-1])
        # # This formula for the advantage comes from:
        # # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        # traj["int_adv"] = discount(delta_t, gamma * lambda_)
        # traj["vf_int_target"] = (
        #     traj["int_adv"] +
        #     traj["vf_int_pred"]).copy().astype(np.float32)


    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS],
             np.array([last_r])])
        discounted_returns = discount(rewards_plus_v,
                                      gamma)[:-1].copy().astype(np.float32)
        
        # # RND
        # int_rewards_plus_v = np.concatenate(
        #     [rollout["int_reward"],
        #      np.array([0.])])
        # int_discounted_returns = discount(rewards_plus_v,
        #                               gamma)[:-1].copy().astype(np.float32)

        if use_critic:
            traj[Postprocessing.
                 ADVANTAGES] = discounted_returns - rollout[SampleBatch.
                                                            VF_PREDS]
            traj[Postprocessing.VALUE_TARGETS] = discounted_returns

            # RND
            # traj["int_adv"] = discounted_returns - rollout["vf_int_pred"]
            # traj["vf_int_target"] = discounted_returns

        else:
            traj[Postprocessing.ADVANTAGES] = discounted_returns
            traj[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                traj[Postprocessing.ADVANTAGES])

            # RND
            # traj["int_adv"] = discounted_returns
            # traj["vf_int_pred"] = np.zeros_like(
            #     traj["int_adv"])

    traj[Postprocessing.ADVANTAGES] = traj[
        Postprocessing.ADVANTAGES].copy().astype(np.float32)

    # traj["int_adv"] = traj["int_adv"].copy().astype(np.float32)

    assert all(val.shape[0] == trajsize for val in traj.values()), \
        "Rollout stacked incorrectly!"
    return SampleBatch(traj)

def postprocess_ppo_gae(policy,
                        sample_batch,
                        other_agent_batches=None,
                        episode=None):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    completed = sample_batch["dones"][-1]
    if completed:
        last_r = 0.0
    else:
        next_state = []
        for i in range(policy.num_state_tensors()):
            next_state.append(sample_batch["state_out_{}".format(i)][-1])
        last_r = policy._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                               sample_batch[SampleBatch.ACTIONS][-1],
                               sample_batch[SampleBatch.REWARDS][-1],
                               *next_state)
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"])
    return batch

def get_policy_class(config):
    return PPORNDTorchPolicy

PPORNDTorchPolicy = build_torch_policy(
    name="PPORNDTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.ppo.ppo.DEFAULT_CONFIG,
    loss_fn=ppo_rnd_surrogate_loss,
    stats_fn=kl_and_loss_stats,
    extra_action_out_fn=vf_preds_fetches,
    postprocess_fn=postprocess_ppo_gae,
    extra_grad_process_fn=apply_grad_clipping,
    before_init=setup_config,
    after_init=setup_mixins,
    mixins=[
        LearningRateSchedule, EntropyCoeffSchedule, KLCoeffMixin,
        ValueNetworkMixin
    ])

PPORNDTrainer = build_trainer(
    name="PPO-RND",
    default_config=DEFAULT_CONFIG,
    default_policy=PPORNDTorchPolicy,
    execution_plan=execution_plan,
    validate_config=validate_config)

