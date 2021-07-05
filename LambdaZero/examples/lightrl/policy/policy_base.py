import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

from LambdaZero.examples.lightrl.a2c_ppo_acktr.distributions import Bernoulli, DiagGaussian
from LambdaZero.examples.lightrl.a2c_ppo_acktr.distributions import FixedCategorical


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

    def forward(self, x):
        return FixedCategorical(logits=x)


class Policy(nn.Module):
    def __init__(self, cfg: Namespace, obs_shape, action_space, base):
        super(Policy, self).__init__()

        self._sample_best = getattr(cfg, "sample_best", False)

        self.base = base

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def post_process_actor_features(self, inputs, actor_features):
        # Mask unavailable actions
        actor_features[~inputs.action_mask] = -20
        return actor_features

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        actor_features = self.post_process_actor_features(inputs, actor_features)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if self._sample_best:
            for ix, i_seed in enumerate(inputs.seed):
                if np.random.rand() > 0.7:
                    smi = self.seed_to_smiles[i_seed.item()]
                    top_r_idx = self.eval_data[smi]["next"]["true_r"].argmax()
                    act_max = self.eval_data[smi]["next"].iloc[top_r_idx].name
                    action[ix] = act_max

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        res_m = Namespace(
            value=value,
            action=action,
            action_log_probs=action_log_probs,
            rnn_hxs=rnn_hxs,
            dist=dist
        )
        return res_m

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        actor_features = self.post_process_actor_features(inputs, actor_features)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs