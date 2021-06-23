import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

'''
Alice agent in the ASP framework. Exploratory policy that sets harder and harder goals for Bob to achieve
'''


class Alice(nn.Module):
    def __init__(self, env, p_hidden_layer_sizes=[32, 32], v_hidden_layer_sizes=[32, 32], p_lr=1e-4, v_lr=1e-4):
        super(Alice, self).__init__()
        self.env = env
        ob = env.reset()
        state_dim = ob.shape[0]
        # print(state_dim)
        action_dim = 3

        # ====== Policy network construction ======
        policy_layers = [
            nn.Linear(state_dim, p_hidden_layer_sizes[0]), nn.ReLU()]

        for i in range(len(p_hidden_layer_sizes) - 1):
            policy_layers.append(
                nn.Linear(p_hidden_layer_sizes[i], p_hidden_layer_sizes[i + 1]))
            policy_layers.append(nn.ReLU())

        policy_layers.append(nn.Linear(p_hidden_layer_sizes[-1], action_dim))

        self.policy = nn.Sequential(*policy_layers)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=p_lr)

        # ====== Value network construction ======
        value_layers = []
        value_layers.append(nn.Linear(state_dim, v_hidden_layer_sizes[0]))
        value_layers.append(nn.ReLU())

        for _ in range(len(v_hidden_layer_sizes) - 1):
            value_layers.append(
                nn.Linear(v_hidden_layer_sizes[i], v_hidden_layer_sizes[i + 1]))
            value_layers.append(nn.ReLU())

        value_layers.append(nn.Linear(v_hidden_layer_sizes[-1], 1))

        self.value = nn.Sequential(*value_layers)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=v_lr)

    def collect_trajectory(self):
        ob = torch.tensor(self.env.reset()).float()
        trajectory = []
        # move until full trajectory is collected (up until horizon)
        done = False
        while not done:
            # we don't want gradients associated with this stuff, all we care is the log probabilities for PPO to do optimization
            with torch.no_grad():
                a_logits = self.policy(ob)
                dist = Categorical(logits=a_logits)

                # epsilon greedy policy here? I don't think we need it because we explore based on Bob's experience anyway
                a = dist.sample()
                lp = dist.log_prob(a)

                next_ob, r, done, _ = self.env.step(a)
                next_ob = torch.tensor([next_ob]).float()
                trajectory.append((ob, a, r, next_ob, lp))
                ob = next_ob

        goal = ob
        return trajectory, goal

    def update_ppo(self, trajs, clamp_param):
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_log_probs = []
        for traj, _ in trajs:
            states = [t[0] for t in traj]
            actions = [t[1] for t in traj]
            rewards = [t[2] for t in traj]
            next_states = [t[3] for t in traj]
            log_probs = [t[4] for t in traj]

            all_states += states
            all_actions += actions
            all_rewards += rewards
            all_next_states += next_states
            all_log_probs += log_probs

        all_states = torch.stack(all_states, dim=0)
        all_actions = torch.stack(all_actions, dim=0)
        all_rewards = torch.tensor(all_rewards).float()
        all_next_states = torch.stack(all_next_states, dim=0)
        all_log_probs = torch.stack(all_log_probs, dim=0)

        # print(all_states.shape)
        # print(all_actions.shape)
        # print(all_rewards.shape)
        # print(all_next_states.shape)
        # print(all_log_probs.shape)
        # print('--------------')

        advs = all_rewards + \
            self.value(all_next_states) - self.value(all_states)

        new_logits = self.policy(all_states)
        dist = Categorical(logits=new_logits)
        new_lps = dist.log_prob(all_actions)
        ratio = torch.exp(new_lps - all_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - clamp_param, 1 + clamp_param)

        # you just do elemental product over ratio and advs and then average, then do the clipping, etc.
        ppo_objective = -torch.min((advs * ratio),
                                   (advs * clipped_ratio)).mean()

        self.policy_optimizer.zero_grad()
        ppo_objective.backward()
        self.policy_optimizer.step()

        # now have to fit value function with MSE loss
        values = self.value(all_states).squeeze(1)
        returns = torch.tensor([torch.sum(all_rewards[i:]) for i in range(
            all_rewards.shape[0])])  # undiscounted sum of rewards
        # print(returns.shape)
        # print(values.shape)
        loss_fn = nn.MSELoss()
        value_loss = loss_fn(returns, values)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return -1 * ppo_objective.detach().item(), value_loss.detach().item()


'''
New Alice framework in the ASP setting. This is an Alice agent that updates itself based off of Bob's probabilities of executing the trajectories in question 
'''


class NewAlice(nn.Module):
    def __init__(self, env, p_hidden_layer_sizes=[32, 32], p_lr=1e-4):
        super(NewAlice, self).__init__()
        self.env = env
        ob = env.reset()
        state_dim = ob.shape[0]
        # print(state_dim)
        action_dim = 3

        # ====== Policy network construction ======
        policy_layers = [
            nn.Linear(state_dim, p_hidden_layer_sizes[0]), nn.ReLU()]

        for i in range(len(p_hidden_layer_sizes) - 1):
            policy_layers.append(
                nn.Linear(p_hidden_layer_sizes[i], p_hidden_layer_sizes[i + 1]))
            policy_layers.append(nn.ReLU())

        policy_layers.append(nn.Linear(p_hidden_layer_sizes[-1], action_dim))

        self.policy = nn.Sequential(*policy_layers)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=p_lr)

    def collect_trajectory(self):
        ob = torch.tensor(self.env.reset()).float()
        trajectory = []
        # move until full trajectory is collected (up until horizon)
        done = False
        while not done:
            # we don't want gradients associated with this stuff, all we care is the log probabilities for PPO to do optimization
            with torch.no_grad():
                a_logits = self.policy(ob)
                dist = Categorical(logits=a_logits)

                # epsilon greedy policy here? I don't think we need it because we explore based on Bob's experience anyway
                a = dist.sample()
                lp = dist.log_prob(a)

                next_ob, r, done, _ = self.env.step(a)
                next_ob = torch.tensor([next_ob]).float()
                trajectory.append((ob, a, r, next_ob, lp))
                ob = next_ob

        goal = ob
        return trajectory, goal

    def update(self, a_trajs, bob_probs):
        # here, bob_probs are the probabilities, for all trajectories, that Bob executes that specific trajectory

        # this is sum of all rewards for all trajectories given, this will increase over time
        rewards = [sum([t[2] for t in traj]) for (traj, _) in a_trajs]
        rewards = torch.tensor(rewards, requires_grad=True).float()
        # print(rewards)

        bob_probs = bob_probs + 1e-8  # just so we're not dividing by 0
        # print(bob_probs)

        objective = -1 * torch.div(rewards, bob_probs).mean()

        self.policy_optimizer.zero_grad()
        objective.backward()
        self.policy_optimizer.step()

        return -1 * objective.detach().item()


class GFlowNetAlice(nn.Module):
    def __init__(self, env, q_hidden_layer_sizes=[32, 32], q_lr=1e-4, c_reg=1e-10):
        super(GFlowNetAlice, self).__init__()
        self.env = env
        state_dim = env.reset().shape[0]
        # print(state_dim)
        action_dim = 3

        self.c_reg = c_reg

        # ====== Policy network construction ======
        q_network_layers = [
            nn.Linear(state_dim, q_hidden_layer_sizes[0]), nn.ReLU()]

        for i in range(len(q_hidden_layer_sizes) - 1):
            q_network_layers.append(
                nn.Linear(q_hidden_layer_sizes[i], q_hidden_layer_sizes[i + 1]))
            q_network_layers.append(nn.ReLU())

        q_network_layers.append(nn.Linear(q_hidden_layer_sizes[-1], action_dim))

        self.q_network = nn.Sequential(*q_network_layers)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)

    def collect_trajectory(self):
        ob = torch.tensor(self.env.reset()).float()
        trajectory = []
        # move until full trajectory is collected (up until horizon)
        done = False
        while not done:
            with torch.no_grad():
                q_values = self.q_network(ob)
                dist = Categorical(logits=q_values)
                a = dist.sample()
                lp = dist.log_prob(a)

                next_ob, r, done, _ = self.env.step(a)
                next_ob = torch.tensor([next_ob]).float()
                trajectory.append((ob, a, r, next_ob, lp))
                ob = next_ob

        goal = ob
        return trajectory, goal

    def update(self, trajectory):
        # print(len(trajectory))
        # if len(trajectory) == 1:
        #     print(trajectory)
        states = [t[0] for t in trajectory]
        actions = [t[1] for t in trajectory]
        rewards = [t[2] for t in trajectory]
        next_states = [t[3] for t in trajectory]

        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack(next_states, dim=0)

        q_values = self.q_network(states)

        inflow_qs = []
        for i in range(len(actions)):
            # takes the Q value of the particular action that was there
            inflow_qs.append(torch.exp(q_values[i][actions[i]]))

        # print(inflow_qs)
        s = sum(inflow_qs)
        inflow = torch.log(self.c_reg + s)

        next_actions = actions[1:]

        next_q_values = self.q_network(next_states)
        outflow_qs = []
        new_rewards = rewards[1:]  # only care about s' rewards
        for i in range(len(next_actions)):
            # same thing, but for Q values for (s', a')
            if i != len(next_actions) - 1:
                outflow_qs.append(
                    new_rewards[i] + torch.exp(next_q_values[i]).sum(-1))
            else:
                outflow_qs.append(new_rewards[i])

        # print(outflow_qs)
        if len(next_actions) == 0:
            s_prime = torch.tensor(rewards[-1])
            # print(s_prime)
        else:
            s_prime = sum(outflow_qs)
        outflow = torch.log(self.c_reg + s_prime)

        loss_fn = nn.MSELoss()
        loss = loss_fn(inflow, outflow)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        return loss.detach().item()


'''
Bob agent in the ASP framework. Essentially a behavioral cloning policy that works to imitate the trajectories laid out by Alice (which progressively leads to higher and higher reward)
'''


class Bob(nn.Module):
    def __init__(self, env, hidden_layer_sizes=[32, 32], lr=1e-4):
        super(Bob, self).__init__()
        self.env = env
        ob = env.reset()
        state_dim = ob.shape[0]
        action_dim = 3

        # ====== Policy network initialization ======
        policy_layers = []
        policy_layers.append(nn.Linear(state_dim, hidden_layer_sizes[0]))

        for i in range(len(hidden_layer_sizes) - 1):
            policy_layers.append(nn.ReLU())
            policy_layers.append(
                nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1]))

        policy_layers.append(nn.ReLU())
        policy_layers.append(nn.Linear(hidden_layer_sizes[-1], action_dim))

        self.policy = nn.Sequential(*policy_layers)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.grad_norm = None

    def collect_trajectory(self, goal_state):
        ob = torch.tensor(self.env.reset()).float()

        trajectory = []

        # move until full trajectory is collected (up until horizon)
        done = False
        while not done:
            # we don't want gradients associated with this stuff, all we care is the log probabilities for PPO to do optimization
            with torch.no_grad():
                a_logits = self.policy(ob)
                dist = Categorical(logits=a_logits)

                # epsilon greedy policy here? I don't think we need it because we explore based on Bob's experience anyway
                a = dist.sample()

                next_ob, r, done, _ = self.env.step(a)
                next_ob = torch.tensor([next_ob]).float()
                trajectory.append((ob, a, r, next_ob))
                ob = next_ob

        reached = (ob == goal_state)
        return trajectory, reached

    def get_traj_probs(self, trajs):
        probs = []
        for traj, _ in trajs:
            states = torch.stack([t[0] for t in traj], dim=0)
            actions = torch.stack([t[1] for t in traj], dim=0)

            logits = self.policy(states)
            dists = Categorical(logits=logits)
            log_probs = dists.log_prob(actions)
            total_prob = torch.exp(torch.sum(log_probs))
            probs.append(total_prob)

        return torch.tensor(probs, requires_grad=True).float()

    def update(self, trajs):
        all_states = []
        all_actions = []

        for traj, _ in trajs:
            states = [t[0] for t in traj]
            actions = [t[1] for t in traj]

            all_states += states
            all_actions += actions

        all_states = torch.stack(all_states, dim=0)
        all_actions = torch.stack(all_actions, dim=0)

        a_logits = self.policy(all_states)
        dists = Categorical(logits=a_logits)
        log_probs = dists.log_prob(all_actions)

        loss = -log_probs.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.grad_norm = torch.sqrt(
            sum([torch.norm(param.grad) for param in self.policy.parameters()]))

        return loss.detach().item()


class GFlowNetBob(nn.Module):
    def __init__(self, env, q_hidden_layer_sizes=[32, 32], q_lr=1e-4, c_reg=1e-10):
        super(GFlowNetBob, self).__init__()
        self.env = env
        state_dim = env.reset().shape[0]
        # print(state_dim)
        action_dim = 3

        self.c_reg = c_reg

        # ====== Policy network construction ======
        q_network_layers = [
            nn.Linear(state_dim, q_hidden_layer_sizes[0]), nn.ReLU()]

        for i in range(len(q_hidden_layer_sizes) - 1):
            q_network_layers.append(
                nn.Linear(q_hidden_layer_sizes[i], q_hidden_layer_sizes[i + 1]))
            q_network_layers.append(nn.ReLU())

        q_network_layers.append(nn.Linear(q_hidden_layer_sizes[-1], action_dim))

        self.q_network = nn.Sequential(*q_network_layers)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=q_lr)

        self.grad_norm = None

    def collect_trajectory(self, goal_state):
        ob = torch.tensor(self.env.reset()).float()
        trajectory = []
        # move until full trajectory is collected (up until horizon)
        done = False
        while not done:
            with torch.no_grad():
                q_values = self.q_network(ob)
                dist = Categorical(logits=q_values)
                a = dist.sample()
                lp = dist.log_prob(a)

                next_ob, r, done, _ = self.env.step(a)
                next_ob = torch.tensor([next_ob]).float()
                trajectory.append((ob, a, r, next_ob, lp))
                ob = next_ob

        bob_reached = (ob == goal_state)
        return trajectory, bob_reached

    def update(self, trajectory):
        # print(len(trajectory))
        # if len(trajectory) == 1:
        #     print(trajectory)
        states = [t[0] for t in trajectory]
        actions = [t[1] for t in trajectory]
        rewards = [t[2] for t in trajectory]
        next_states = [t[3] for t in trajectory]

        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack(next_states, dim=0)

        q_values = self.q_network(states)

        inflow_qs = []
        for i in range(len(actions)):
            # takes the Q value of the particular action that was there
            inflow_qs.append(torch.exp(q_values[i][actions[i]]))

        # print(inflow_qs)
        s = sum(inflow_qs)
        inflow = torch.log(self.c_reg + s)

        next_actions = actions[1:]

        next_q_values = self.q_network(next_states)
        outflow_qs = []
        new_rewards = rewards[1:]  # only care about s' rewards
        for i in range(len(next_actions)):
            # same thing, but for Q values for (s', a')
            if i != len(next_actions) - 1:
                outflow_qs.append(
                    new_rewards[i] + torch.exp(next_q_values[i]).sum(-1))
            else:
                outflow_qs.append(new_rewards[i])

        # print(outflow_qs)
        if len(next_actions) == 0:
            s_prime = torch.tensor(rewards[-1])
            # print(s_prime)
        else:
            s_prime = sum(outflow_qs)
        outflow = torch.log(self.c_reg + s_prime)

        loss_fn = nn.MSELoss()
        loss = loss_fn(inflow, outflow)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

        # gradient of loss with respect to the Q network parameters--but to take the norm it's a little weird
        self.grad_norm = torch.sqrt(sum([torch.norm(param.grad)
                                         for param in self.q_network.parameters()]))

        return loss.detach().item()
