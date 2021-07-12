import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])



class ProcessList:
    def __init__(self, lst, process):
        self._lst = lst
        self._process = process

    def __getitem__(self, item):
        return self._process(self._lst[item])


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, process_obs):

        self._obs = [None] * (num_steps + 1)
        self.obs = ProcessList(self._obs, process_obs)

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_processes, recurrent_hidden_state_size)
        self.rewards = [torch.zeros(num_steps, num_processes, 1) for _ in range(2)]
        self.value_preds = [torch.zeros(num_steps + 1, num_processes, 1) for _ in range(2)]
        self.returns = [torch.zeros(num_steps + 1, num_processes, 1) for _ in range(2)]
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.process_obs = process_obs

        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        # Masks that indicate whether it's a true terminal state
        # or time limit end state
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        # self.obs = self.obs.to(device)
        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.rewards = [x.to(device) for x in self.rewards]
        self.value_preds = [x.to(device) for x in self.value_preds]
        self.returns = [x.to(device) for x in self.returns]
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)

    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks):
        self._obs[self.step + 1] = obs
        self.recurrent_hidden_states[self.step + 1].copy_(recurrent_hidden_states)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        for i in range(2):
            self.value_preds[i][self.step].copy_(value_preds[i])
            self.rewards[i][self.step].copy_(rewards[i])
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        self.step = (self.step + 1) % self.num_steps


    def after_update(self):
        self._obs[0] = self._obs[-1]
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self,
                        next_value,
                        use_gae,
                        gamma,
                        gae_lambda,
                        use_proper_time_limits=True):
        for i in range(2):
            value_preds = self.value_preds[i]
            rewards = self.rewards[i]
            returns = self.returns[i]
            next_v = next_value[:, i].unsqueeze(-1)

            if use_proper_time_limits:
                if use_gae:
                    value_preds[-1] = next_v
                    gae = 0
                    for step in reversed(range(rewards.size(0))):
                        delta = rewards[step] + gamma * value_preds[
                            step + 1] * self.masks[step +
                                                   1] - value_preds[step]
                        gae = delta + gamma * gae_lambda * self.masks[step +
                                                                      1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        returns[step] = gae + value_preds[step]
                else:
                    returns[-1] = next_v
                    for step in reversed(range(rewards.size(0))):
                        returns[step] = (returns[step + 1] * \
                            gamma * self.masks[step + 1] + rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_preds[step]
            else:
                if use_gae:
                    value_preds[-1] = next_v
                    gae = 0
                    for step in reversed(range(rewards.size(0))):
                        delta = rewards[step] + gamma * value_preds[step + 1] * self.masks[step + 1] - value_preds[step]
                        gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                        returns[step] = gae + value_preds[step]
                else:
                    returns[-1] = next_v
                    for step in reversed(range(rewards.size(0))):
                        returns[step] = returns[step + 1] * \
                            gamma * self.masks[step + 1] + rewards[step]

    def feed_forward_generator(self,
                               advantages,
                               num_mini_batch=None,
                               mini_batch_size=None):
        num_steps, num_processes = self.rewards[0].size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True)

        flat_obs = [item for sublist in self._obs for item in sublist]
        process_obs = self.process_obs
        for indices in sampler:
            obs_batch = process_obs([flat_obs[x] for x in indices])
            recurrent_hidden_states_batch = self.recurrent_hidden_states[:-1].view(
                -1, self.recurrent_hidden_states.size(-1))[indices]
            actions_batch = self.actions.view(-1,
                                              self.actions.size(-1))[indices]
            value_preds_batch = [vvv[:-1].view(-1, 1)[indices] for vvv in self.value_preds]
            return_batch = [vvv[:-1].view(-1, 1)[indices] for vvv in self.returns]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1,
                                                                    1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def recurrent_generator(self, advantages, num_mini_batch):
        raise NotImplementedError

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch))
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            obs_batch = []
            recurrent_hidden_states_batch = []
            actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                obs_batch.append(self._obs[:-1, ind])
                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(
                    self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            T, N = self.num_steps, num_envs_per_batch
            # These are all tensors of size (T, N, -1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            old_action_log_probs_batch = torch.stack(
                old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            obs_batch = _flatten_helper(T, N, obs_batch)
            actions_batch = _flatten_helper(T, N, actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            return_batch = _flatten_helper(T, N, return_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)
            old_action_log_probs_batch = _flatten_helper(T, N, \
                    old_action_log_probs_batch)
            adv_targ = _flatten_helper(T, N, adv_targ)

            yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ