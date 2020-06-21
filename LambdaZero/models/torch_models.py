import os,time
from abc import ABC
import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from ray.rllib.models.tf.misc import normc_initializer

th, nn = try_import_torch()
from torch_geometric.nn import NNConv, Set2Set

from LambdaZero.utils import RunningMeanStd

def convert_to_tensor(arr):
    tensor = th.from_numpy(np.asarray(arr))
    if tensor.dtype == th.double:
        tensor = tensor.float()
    return tensor

class ActorCriticModel(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)

        self.shared_layers = None
        self.actor_layers = None
        self.critic_layers = None

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.shared_layers(x)
        # actor outputs
        logits = self.actor_layers(x)
        # compute value
        self._value_out = self.critic_layers(x)
        return logits, None

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        obs = convert_to_tensor([self.preprocessor.transform(obs)])
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with th.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = th.squeeze(logits), th.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value



class MolActorCritic_thv1(TorchModelV2, nn.Module, ABC):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)
        mol_fp_len = obs_space.original_space["mol_fp"].shape[0]
        stem_fp_len = obs_space.original_space["stem_fps"].shape[1]
        n_blocks = (obs_space.original_space["action_mask"].shape[0] -
                    obs_space.original_space["jbond_fps"].shape[0] - 1) // obs_space.original_space["stem_fps"].shape[0]
        self.max_steps = obs_space.original_space["num_steps"].n

        self.shared_layers = nn.Sequential(
            nn.Linear(in_features=mol_fp_len + self.max_steps, out_features=256),nn.ReLU(),
            nn.Linear(in_features=256, out_features=256))
        self.stem_layers = nn.Sequential(
            nn.Linear(in_features=stem_fp_len, out_features=256))

        # build actor
        self.proc_stem = nn.Sequential(nn.Linear(in_features=stem_fp_len, out_features=256))
        self.actor_stop = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256, out_features=1))
        self.actor_add = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256*3, out_features=n_blocks))
        self.actor_break = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256*3, out_features=n_blocks))
        # build critic
        self.critic_layers = nn.Sequential(nn.ReLU(), nn.Linear(in_features=256, out_features=1))
        self._value_out = None
        
        # RND
        if "rnd_weight" in kwargs.keys():
            self.rnd_weight = kwargs["rnd_weight"]
        else: 
            self.rnd_weight = 0

        if self.rnd_weight > 0:
            self.rnd_target = nn.Sequential(
                nn.Linear(in_features=mol_fp_len, out_features=256),nn.ReLU(),
                nn.Linear(in_features=256, out_features=256), nn.ReLU(),
                nn.Linear(in_features=265, out_features=1))
            self.rnd_predictor = nn.Sequential(
                nn.Linear(in_features=mol_fp_len, out_features=256),nn.ReLU(),
                nn.Linear(in_features=256, out_features=256), nn.ReLU(),
                nn.Linear(in_features=265, out_features=1))
            self.rnd_stats = RunningMeanStd(shape=(mol_fp_len))
            # Freeze target network
            self.rnd_target.eval()
            for param in self.rnd_target.parameters():
                param.requires_grad = False


    def forward(self, input_dict, state, seq_lens):
        # shared molecule embedding
        # weak todo (maksym) use mask before compute
        obs = input_dict # ['obs']
        mol_fp = obs["mol_fp"]
        stem_fps = obs["stem_fps"]
        jbond_fps = obs["jbond_fps"]
        num_steps = obs["num_steps"]
        action_mask = obs["action_mask"]

        # shared layers
        mol_embed = self.shared_layers(th.cat([mol_fp, num_steps], 1))

        # actor outputs
        stop_logit = self.actor_stop(mol_embed)
        #
        jbond_embed = self.stem_layers(jbond_fps)
        mol_embed_ = mol_embed[:,None, :].repeat([1, jbond_embed.shape[1], 1])
        jbond_embed = th.cat([jbond_embed, mol_embed_, jbond_embed * mol_embed_], dim=2)
        break_logits = self.actor_break(jbond_embed)[:,:,0]
        #
        stem_embed = self.stem_layers(stem_fps)
        mol_embed_ = mol_embed[:,None, :].repeat([1, stem_embed.shape[1], 1])
        stem_embed = th.cat([stem_embed, mol_embed_, stem_embed * mol_embed_], dim=2)
        add_logits = self.actor_add(stem_embed)
        add_logits = add_logits.reshape([add_logits.shape[0], -1])
        #
        actor_logits = th.cat([stop_logit, break_logits, add_logits], axis=1)
        # compute value
        critic_logits = self.critic_layers(mol_embed)
        self._value_out = critic_logits[:,0]

        # mask not available actions
        masked_actions = (1. - action_mask).to(th.bool)
        actor_logits[masked_actions] = -20 # some very small prob that does not lead to inf
        return actor_logits, state

    def value_function(self):
        return self._value_out

    def compute_priors_and_value(self, obs):
        obs = th.tensor([self.preprocessor.transform(obs)]).float().cuda()
        input_dict = restore_original_dimensions(obs, self.obs_space, "torch")

        with th.no_grad():
            model_out = self.forward(input_dict, None, [1])
            logits, _ = model_out
            value = self.value_function()
            logits, value = th.squeeze(logits), th.squeeze(value)
            priors = nn.Softmax(dim=-1)(logits)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()
            return priors, value
        
    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        if self.rnd_weight > 0:
            obs = ((loss_inputs['obs']['mol_fp'] - self.rnd_stats.mean) / (np.sqrt(obs_rms.var))).clip(0, 10)
            target_reward = self.rnd_target(obs)
            predictor_reward = self.rnd_predictor_reward(obs)
            rnd_loss = ((target_reward - predictor_reward) ** 2).sum()
            self.rnd_stats.update(loss_inputs['obs']['mol_fp'])
            return policy_loss + self.rnd_weight * rnd_loss
        else:
            return policy_loss
        
    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        th.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(th.load(checkpoint_path))


class MPNNet(th.nn.Module):
    """
    A message passing neural network implementation based on Gilmer et al. <https://arxiv.org/pdf/1704.01212.pdf>
    """
    def __init__(self, num_feat=14, dim=64):
        super(MPNNet, self).__init__()
        self.lin0 = th.nn.Linear(num_feat, dim)

        h = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, h, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = th.nn.Linear(2 * dim, dim)
        self.lin2 = th.nn.Linear(dim, 1)

    def forward(self, data):
        out = nn.functional.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = nn.functional.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = nn.functional.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)
