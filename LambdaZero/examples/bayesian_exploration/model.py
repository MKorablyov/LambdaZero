from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, valid_padding, \
    SlimConv2d, SlimFC
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn
from ray.rllib.utils import try_import_torch
import numpy as np

import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn

# from blitz.module import BayesianLinear
from LambdaZero.utils import RunningMeanStd

torch, nn = try_import_torch()


class BayesianVisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, **kw):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        activation = get_activation_fn(
            model_config.get("conv_activation"), framework="torch")
        filters = model_config.get("conv_filters")
        if not filters:
            filters = _get_filter_config(obs_space.shape)
        # no_final_linear = model_config.get("no_final_linear")
        # vf_share_layers = model_config.get("vf_share_layers")

        layers = []
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = valid_padding(in_size, kernel,
                                              [stride, stride])
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,
                activation_fn=activation))
        self._convs = nn.Sequential(*layers)
        self.rnd_weight = kw.get("rnd_weight", 0)
        self.rnd = (self.rnd_weight != 0)
        rnd_output_dim = kw.get("rnd_output_dim", 1)
        self.rnd_adv_weight = kw.get("rnd_adv_weight", 1.0)
        self.rnd_vf_loss_weight = kw.get("rnd_vf_loss_weight", 1.0)

        # self.rnd_weight = model_config['custom_model_config'].get("rnd_weight", 0)
        # self.rnd = (self.rnd_weight != 0)
        # rnd_output_dim = model_config['custom_model_config'].get("rnd_output_dim", 1)
        # self.rnd_adv_weight = model_config['custom_model_config'].get("rnd_adv_weight", 1.0)
        # self.rnd_vf_loss_weight = model_config['custom_model_config'].get("rnd_vf_loss_weight", 1.0)

        self._logits = SlimFC(
            out_channels, num_outputs, initializer=nn.init.xavier_uniform_)
        self._value_branch = SlimFC(
            out_channels, 1, initializer=normc_initializer())
        # Holds the current "base" output (before logits layer).
        self._features = None

        if self.rnd:
            target_layers = []
            (w, h, in_channels) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = valid_padding(in_size, kernel,
                                                [stride, stride])
                target_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation))
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            target_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation))
            target_layers.append(nn.Flatten())
            target_layers.append(SlimFC(
                out_channels, rnd_output_dim, initializer=normc_initializer()))
            self.rnd_target = nn.Sequential(*target_layers)

            pred_layers = []
            (w, h, in_channels) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = valid_padding(in_size, kernel,
                                                [stride, stride])
                pred_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation))
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            pred_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation))
            pred_layers.append(nn.Flatten())
            pred_layers.append(SlimFC(
                out_channels, rnd_output_dim, initializer=normc_initializer()))
            self.rnd_pred = nn.Sequential(*pred_layers)

            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.orthogonal_(m.weight, np.sqrt(2))
                    m.bias.data.zero_()

            for param in self.rnd_target.parameters():
                param.requires_grad = False

            self._value_int = SlimFC(
                out_channels, 1, initializer=normc_initializer())
            self.rnd_obs_stats = RunningMeanStd(shape=obs_space.shape)
            self.rnd_rew_stats = RunningMeanStd(shape=())

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        self._features = self._hidden_layers(input_dict["obs"].float())
        logits = self._logits(self._features)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value_branch(self._features).squeeze(1)

    def int_value_function(self):
        assert self._features is not None, "must call forward() first"
        return self._value_int(self._features).squeeze(1)

    def compute_intrinsic_rewards(self, train_batch):
        obs_ = train_batch['new_obs']
        self.rnd_obs_stats.update(obs_.clone().detach().cpu().numpy())
        # import pdb; pdb.set_trace();
        obs_mean = torch.as_tensor(self.rnd_obs_stats.mean, dtype=torch.float32, device=train_batch['obs'].device)
        obs_var = torch.as_tensor(self.rnd_obs_stats.var, dtype=torch.float32, device=train_batch['obs'].device)
        obs = ((obs_ - obs_mean) / (torch.sqrt(obs_var))).clamp(-5, 5)
        
        target_reward = self.rnd_target(obs.permute(0, 3, 1, 2))
        predictor_reward = self.rnd_pred(obs.permute(0, 3, 1, 2))
        rnd_loss_ = ((target_reward - predictor_reward) ** 2).sum(1) / 2
        self.rnd_rew_stats.update(rnd_loss_.clone().detach().cpu().numpy())
        
        rew_mean = torch.as_tensor(self.rnd_rew_stats.mean, dtype=torch.float32, device=rnd_loss_.device)
        rew_var = torch.as_tensor(self.rnd_rew_stats.var, dtype=torch.float32, device=rnd_loss_.device)
        rnd_loss = (rnd_loss_ - rew_mean) / (torch.sqrt(rew_var))
        
        return rnd_loss

    def _hidden_layers(self, obs):
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class RNDModel(nn.Module):
    def __init__(self):
        super(RNDModel, self).__init__()
        # self.input_size = input_size
        # self.output_size = output_size
        feature_output = 7 * 7 * 64

        self.target = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512)
        )

        self.predictor = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        # Initialize weights    
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.orthogonal_(m.weight, np.sqrt(2))
                m.bias.data.zero_()

        # Set target parameters as untrainable
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature