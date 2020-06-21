import sys, os, time, socket
import numpy as np
import copy
from scipy import special
from sklearn.gaussian_process import GaussianProcessRegressor
import torch
from rdkit import Chem
from matplotlib import pyplot as plt
import json
import pandas
import itertools

import ray
from ray import tune

import LambdaZero.utils

from LambdaZero.environments.block_mol_v5 import BlockMolEnv_v5
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as DEFAULT_ENV_CONFIG

from LambdaZero.examples.mf_gp_ucb.replay_buffer import ReplayBuffer
from LambdaZero.examples.mf_gp_ucb.model import ValNetwork

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

class TrainableRandom(tune.Trainable):
    def _setup(self, config):
        # Environment
        c = DEFAULT_ENV_CONFIG
        c["reward_config"]["device"] = "cpu"
        self.env = BlockMolEnv_v5(c)
        self.done = True
        # Reward function
        reward = LambdaZero.environments.reward.PredDockReward(load_model=config['dockscore_model'],
                                natm_cutoff=[45, 50],
                                qed_cutoff=[0.2, 0.7],
                                soft_stop=False,
                                exp=None,
                                delta=False,
                                simulation_cost=0.0,
                                device="cpu")
        db_name = "actor_dock"
        docksetup_dir = os.path.join(datasets_dir, "brutal_dock/d4/docksetup")
        #docksetup_dir = os.path.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup"),
        dock_smi = LambdaZero.chem.Dock_smi(outpath=os.path.join(datasets_dir, db_name, "dock"),
                         chimera_dir=os.path.join(programs_dir, "chimera"),
                         dock6_dir=os.path.join(programs_dir, "dock6"),
                         docksetup_dir=docksetup_dir,
                         gas_charge=True)
        def oracle1(mol):
            reward.reset()
            return reward(mol,env_stop=False,simulate=True,num_steps=1)[0]
        def oracle2(mol):
            try:
                name, energy, coord = dock_smi.dock(Chem.MolToSmiles(mol.mol))
                return energy
            except AssertionError as e:
                # AssertionError: parsing error - multiple gridscores
                # Dock6 fails sometimes, and those failure are most likely on very high energy molecules
                return 0
        self.oracles = [
            oracle2
        ]
        # Store all visited states and their values
        self.values = []
        self.states = []
    def evaluate_molecules(self, molecules):
        return [self.oracles[0](mol) for mol in molecules]
    def save_current_state(self):
        mol = self.info['molecule']
        self.states.append(self.env.get_state())
        self.values.append(self.oracles[0](mol))
    def step(self):
        if self.done:
            self.obs = self.env.reset()
        actions = np.where(self.obs['action_mask'])[0]
        action = np.random.choice(actions.flatten())
        self.obs, _, self.done, self.info = self.env.step(action)
        # Save visited states and their values
        self.save_current_state()
    def get_top_k(self, top_k):
        """ Return the the top k molecules and their scores """
        print('NUMBER OF MOLECULES',len(self.values))
        if len(self.values) < top_k:
            return [(None,None) for _ in range(top_k)]
        top_idx = np.argsort(self.values)[:top_k]
        return tuple(zip(*[(self.values[i], self.states[i]) for i in top_idx]))
    def _train(self):
        self.step()
        top_k = self.get_top_k(1)
        return {'score': top_k[0]}
    def _save(self, chkpt_dir):
        return {}

class TrainableBoltzmann(TrainableRandom):
    def step(self):
        if self.done:
            self.obs = self.env.reset()

        rng = np.random
        env = self.env
        state = env.get_state()
        action_mask = self.obs['action_mask']
        temperature = 1
        evaluate_molecules = self.evaluate_molecules

        # Copied from Clement's code >>>>>
        actions = np.where(action_mask)[0]
        molecules = []
        next_states = []
        observations = []

        for i, a in enumerate(actions):
            env.set_state(state)
            obs, _, _, info = env.step(a)
            molecules.append(copy.deepcopy(info["molecule"]))
            next_states.append(env.get_state())
            observations.append(obs)

        values = evaluate_molecules(molecules)
        values = -np.array(values)
        probs = special.softmax(np.divide(values, temperature))

        a = rng.choice(actions.shape[0], p=probs)
        # <<<< Clement's code ends here

        action = actions[a]
        env.set_state(state)
        self.obs, _, self.done, self.info = self.env.step(action)
        self.save_current_state()

class TrainableMFGPUCB(TrainableRandom):
    def _setup(self, config):
        super()._setup(config)
        self.device = config['device']

        self.replay_buffer = ReplayBuffer(config['replay_buffer_size'])
        self.warmup_steps = config['warmup_steps']
        self.sv_net_exploit = ValNetwork(config['net_structure']).to(self.device)
        self.sv_net_explore = ValNetwork(config['net_structure']).to(self.device)
        self.sv_net_exploit_target = ValNetwork(config['net_structure']).to(self.device)
        self.sv_net_explore_target = ValNetwork(config['net_structure']).to(self.device)
        self.optim_explore = torch.optim.Adam(self.sv_net_explore.parameters(), lr=config['learning_rate'])
        self.optim_exploit = torch.optim.Adam(self.sv_net_exploit.parameters(), lr=config['learning_rate'])
        self.polyak_rate = 1e-3
        self.evaluation_criterion = 'kandasamy'
        #self.evaluation_criterion = 'always'
        self.oracle_data = None
        self.estimates = [GaussianProcessRegressor() for _ in self.oracles]

        self.zeta = config['zeta']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.last_query = [0]*len(self.oracles) # Number of iterations since each oracle was queried
    def neighbours(self, state, include_actions=False):
        env = self.env
        state = env.get_state()
        action_mask = self.state_to_obs(state)['action_mask']
        actions = np.where(action_mask)[0]

        output = []
        for a in actions:
            env.set_state(state)
            obs, _, _, info = env.step(a)
            if include_actions:
                output.append((a,obs['mol_fp']))
            else:
                output.append(obs['mol_fp'])
        env.set_state(state)
        return output
    def is_warming_up(self):
        return len(self.replay_buffer) < self.warmup_steps
    def state_to_obs(self,state):
        s = self.env.get_state()
        obs = self.env.set_state(state)
        self.env.set_state(s)
        return obs
    def state_to_mol(self,state):
        s = self.env.get_state()
        self.env.set_state(state)
        mol = self.env.molMDP.molecule
        self.env.set_state(s)
        return mol
    def compute_state_value_ucb(self,state,fidelity):
        max_fidelity = len(self.oracles)-1
        fidelity_diff = max_fidelity-fidelity
        zeta = self.zeta*fidelity_diff
        gp = self.estimates[fidelity]
        mu,std = gp.predict(self.state_to_obs(state)['mol_fp'].reshape(1,-1),return_std=True)
        beta = self.beta
        phi = mu + (beta**(1/2))*std+zeta
        return phi.item()
    def compute_state_value_explore(self, state):
        """
        state - a single state from env.get_state()
        """
        neighbours = self.neighbours(state)
        if len(neighbours) > 0:
            n = torch.stack([torch.tensor(n) for n in neighbours]).float().to(self.device)
            max_val = self.sv_net_explore_target(n).max().item()
        else:
            max_val = -float('inf')
        phi = [self.compute_state_value_ucb(state,fidelity) for fidelity in range(len(self.oracles))]
        if len(phi) == 1:
            max_phi = phi[0]
        else:
            max_phi = max(*phi)
        return max(max_val,max_phi)
    def step(self):
        self.init_oracle_data()

        if self.done:
            self.obs = self.env.reset()

        if self.is_warming_up():
            actions = np.where(self.obs['action_mask'])[0]
            action = np.random.choice(actions.flatten()) # Random action
            self.obs, _, self.done, self.info = self.env.step(action)
            self.replay_buffer._add_to_buffer((self.obs['mol_fp'], self.env.get_state()))
            return

        def policy(values):
            return torch.distributions.Categorical(logits=-values)
        val_func = self.sv_net_explore

        neighbours = self.neighbours(self.env.get_state(),include_actions=True)
        next_obs = torch.stack([torch.tensor(n[1]) for n in neighbours]).to(self.device)
        vals = val_func(next_obs).flatten().detach()
        dist = policy(vals)
        a = dist.sample().item()
        action = neighbours[a][0]

        self.obs,_,self.done,self.info = self.env.step(action)

        self.evaluate()
        self.train_vals()
    def train_vals(self,batch_size=2,iterations=1):
        if self.is_warming_up():
            return
        if len(self.replay_buffer) < batch_size:
            return
        dataloader = torch.utils.data.DataLoader(
                self.replay_buffer, batch_size=batch_size, shuffle=True)
        tau = self.polyak_rate
        optim_explore = self.optim_explore
        optim_exploit = self.optim_exploit

        # Choose a batch
        random_indices = torch.tensor([range(len(self.replay_buffer))]).flatten()
        batch_indices = random_indices[:batch_size*iterations]
        batch_indices = batch_indices.view(iterations,batch_size)

        # Train
        for bi in batch_indices:
            loss_explore = 0
            #loss_exploit = 0
            for i in bi:
                o,s = self.replay_buffer[i]
                o = torch.tensor(o).to(self.device)
                loss_explore += (self.sv_net_explore(o)-self.compute_state_value_explore(s))**2
                #loss_exploit += (self.sv_net_exploit(o)-self.compute_state_value_exploit(s))**2
            optim_explore.zero_grad()
            #optim_exploit.zero_grad()
            loss_explore.backward()
            #loss_exploit.backward()
            optim_explore.step()
            #optim_exploit.step()

            # Update target weights
            #for p1,p2 in zip(self.sv_net_exploit_target.parameters(), self.sv_net_exploit.parameters()):
            #    p1.data = (1-tau)*p1+tau*p2
            for p1,p2 in zip(self.sv_net_explore_target.parameters(), self.sv_net_explore.parameters()):
                p1.data = (1-tau)*p1+tau*p2
    def evaluate(self):
        """ Call oracle on the current observation if needed. Update the agent's belief and return the runtime."""
        # Check if we're done warmup
        if self.is_warming_up():
            return
        if self.oracle_data is None:
            return
        # Check if obs needs evaluating
        state = self.env.get_state()
        obs = torch.tensor(self.obs['mol_fp']).to(self.device)
        mol = self.info['molecule']
        if self.evaluation_criterion == 'kandasamy':
            needs_evaluation = self.compute_state_value_explore(state) >= self.sv_net_explore(obs)
        elif self.evaluation_criterion == 'always':
            needs_evaluation = True
        else:
            raise Exception('Invalid evaluation criterion %s' % self.evaluation_criterion)
        # Evaluate
        if needs_evaluation:
            # evaluate and return runtime
            for fid in range(len(self.oracles)):
                _,std = self.estimates[fid].predict(obs.reshape(1,-1).cpu(),return_std=True)
                if np.sqrt(self.beta)*std < self.gamma[fid] and fid < len(self.oracles)-1:
                    # If we're reasonably certain of this estimate, move on to a higher fidelity
                    continue
                # Don't evaluate if it's already been evaluated
                # TODO: I don't think fingerprint is unique to a single molecule. Find another key.
                if tuple(obs.tolist()) in self.oracle_data[fid]:
                    continue
                # Evaluate at chosen fidelity
                value = self.oracles[fid](mol)
                self.oracle_data[fid][tuple(obs.tolist())] = -value
                # Update Gaussian processes
                x = list(self.oracle_data[fid].keys())
                y = [self.oracle_data[fid][k] for k in x]
                self.estimates[fid].fit(x,y)
                # Update gamma
                self.update_gamma(fid)
                # Save evluated molecule
                self.states.append(state)
                self.values.append(value)
    def init_oracle_data(self):
        if self.oracle_data is not None:
            return # Don't initialize again
        if len(self.replay_buffer) < self.warmup_steps:
            return
        self.oracle_data = [{} for _ in self.oracles] # Values from oracle calls
        for i in range(len(self.oracles)):
            for x in np.random.choice(list(range(len(self.replay_buffer))),size=5,replace=False):
                obs,state = self.replay_buffer[x]
                mol = self.state_to_mol(state)
                if obs is None: # We'll just start with one less. nbd.
                    continue
                val = self.oracles[i](mol)
                self.oracle_data[i][tuple(obs.tolist())] = -val
                if i == len(self.oracles)-1:
                    self.states.append(state)
                    self.values.append(val)
            x = list(self.oracle_data[i].keys())
            y = [self.oracle_data[i][k] for k in x]
            self.estimates[i].fit(x,y)
    def update_gamma(self, fidelity):
        """ To be called whenever an oracle is called. """
        for i in range(len(self.oracles)):
            self.last_query[i] += 1
        self.last_query[fidelity] = 0
        for i in range(len(self.oracles)-1):
            if self.last_query[i] >= self.oracle_costs[i+1]/self.oracle_costs[i]:
                self.gamma[i] *= 2

DEFAULT_RANDOM_CONFIG = {
    'name': 'DEFAULT_RANDOM',
    'run_or_experiment': TrainableRandom,
    'config': {
        'dockscore_model': '/home/ml/hhuang63/Summaries/BasicRegressor/BasicRegressor_0_2020-06-02_17-41-06bzogful8/checkpoint_200/model.pth'
    },
    "local_dir": summaries_dir,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 100},
    'resources_per_trial': {
        'cpu': 10
    }
}

DEFAULT_BOLTZMANN_CONFIG = {
    'name': 'DEFAULT_BOLTZMANN',
    'run_or_experiment': TrainableBoltzmann,
    'config': {
        'dockscore_model': '/home/ml/hhuang63/Summaries/BasicRegressor/BasicRegressor_0_2020-06-02_17-41-06bzogful8/checkpoint_200/model.pth'
    },
    "local_dir": summaries_dir,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 100},
    'resources_per_trial': {
        'cpu': 10
    }
}

DEFAULT_MFGPUCB_CONFIG = {
    'name': 'DEFAULT_MFGPUCB',
    'run_or_experiment': TrainableMFGPUCB,
    'config': {
        'dockscore_model': '/home/ml/hhuang63/Summaries/BasicRegressor/BasicRegressor_0_2020-06-02_17-41-06bzogful8/checkpoint_200/model.pth',
        'replay_buffer_size': 1000,
        'warmup_steps': 10,
        'net_structure': [512,256,1],
        'learning_rate': 1e-3,
        'zeta': 0.1, # Max difference between oracles
        'gamma': [0.1], # Threshold for deciding between oracles
        'beta': 1, # Standard deviation weighting
        'device': 'cuda',
    },
    "local_dir": summaries_dir,
    "checkpoint_freq": 250,
    "stop":{"training_iteration": 100},
    'resources_per_trial': {
        'cpu': 10,
        'gpu': 0.25
    }
}

def plot(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    dfs = []
    for checkpoint in data['checkpoints']:
        logdir = checkpoint['logdir']
        df = pandas.read_csv(os.path.join(logdir,'progress.csv'))
        dfs.append(df)

    def normalize_time(t,v,stride=1):
        start_time = t[0]
        output_v = []
        output_t = []
        curr_t = start_time
        for v0,v1,t0,t1 in zip(v,v[1:],t,t[1:]):
            a = (v1-v0)/(t1-t0)
            while curr_t <= t1:
                output_v.append(a*(curr_t-t0)+v0)
                output_t.append(curr_t-start_time)
                curr_t += stride
        return output_t,output_v

    def mean(series):
        output = []
        for x in itertools.zip_longest(*series):
            output.append(np.mean(list(filter(lambda a: a is not None, x))))
        return output

    stride = 1
    data = [normalize_time(d.timestamp,d.score.apply(lambda x: eval(x)[0]),stride=stride) for d in dfs]
    y = mean([y for x,y in data])
    t = range(0,len(y)*stride,stride)

    plt.plot(t,y)

if __name__=='__main__':
    config = DEFAULT_RANDOM_CONFIG
    #config = DEFAULT_BOLTZMANN_CONFIG
    #config = DEFAULT_MFGPUCB_CONFIG
    analysis = tune.run(
        max_failures=0,
        num_samples=10,
        **config
    )

    #from matplotlib import pyplot as plt
    #import pandas
    #dfs = analysis.trial_dataframes
    #data = [d.score.apply(lambda x: eval(x)[0]) for d in dfs.values()]
    #ax = pandas.concat(data,axis=1).mean(axis=1).plot()
    #ax.set_xlabel('Iterations')
    #ax.set_ylabel('Binding Energy')
    #ax.grid(which='both')
    #plt.savefig('b.png')

    #plot(os.path.join(summaries_dir, 'DEFAULT_RANDOM', 'experiment_state-2020-06-20_19-28-18.json'))
    #plt.xlabel('Time')
    #plt.ylabel('Binding Energy')
    #plt.grid(which='both')
    #plt.savefig('b.png')
