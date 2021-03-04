import random
import numpy as np
from rdkit import Chem
from gym.spaces import Discrete, Dict
import torch

from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.agents.trainer_template import build_trainer

from LambdaZero.inputs import mol_to_graph


class BlockMoleculeData_wrapper():
    def __init__(self, mol, smiles):
        self.graph = None
        self.mol = mol
        self.smiles = smiles


class HTP_Env_v0(tune.Trainable):

    def setup(self, config=None):
        self.mc_sampling = config["mc_sampling"](config["mc_sampling_config"])
        self.reward = config["reward"](**config["reward_config"])
        print('environment initilization done')

    def step(self):
        for i in range(50): # to not return every single time
            products = None
            while products is None:
                products = self.mc_sampling.sample_mol() # does the number of steps in config
            agent_stop = True
            env_stop = True

            mol = products[0]
            smiles = products[1]
            molecule = BlockMoleculeData_wrapper(mol, smiles)

            try: # for issues in graph_to_mol
                graph = mol_to_graph(smiles)
                graph.x = torch.cat([graph.x, torch.zeros([graph.x.shape[0], 2])], dim=1)
                graph.jbond_atmidx = torch.zeros([0, 2])
                graph.jbond_preds = torch.zeros([0, 0])
                graph.stem_atmidx = torch.zeros([0])
                graph.stem_preds = torch.zeros([0, 0])
                # atmfeat, _, bond, bondfeat = chem.mpnn_feat(mol, ifcoord=False)
                # graph = chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat, data_cls=ParametricMolData)
                molecule.graph = graph

                reward, log_vals = self.reward(molecule, agent_stop, env_stop, None)
            except Exception as e:
                print (e)
                reward = 0.0
                log_vals = {}
            info = {"molecule": smiles, "log_vals": log_vals}
        # done = any((agent_stop, env_stop))
        # print(reward, log_vals)
        return {'reward': reward, **info} # None, reward, done, info


class HTP_Env_v1():
    def __init__(self, config=None):
        self.mc_sampling = config["mc_sampling"](config["mc_sampling_config"])
        self.reward = config["reward"](**config["reward_config"])
        self.max_steps = config["max_steps"]

        self.action_space = Discrete(10, ) # doesnt do anything right now
        self.observation_space = Dict({"step": Discrete(n=self.max_steps+1)}) # doesn't do anything rn

        self.reset()
        print('environment initilization done')

    def reset(self):
        self.num_step = 0
        self.obs = {"step": self.num_step}
        # initialize with a molecule as reactant
        self.molecule = random.choice(self.mc_sampling.mols)
        # print(self.molecule)
        self.molecule = self.mc_sampling.salt_remover(Chem.MolFromSmiles(self.molecule))
        return self.obs

    def step(self, action):
        # evaluates at every step, and loops until a product is made
        product = None
        while product is None:
            reactant = random.choice(self.mc_sampling.mols)
            # print(reactant)
            reactant = self.mc_sampling.salt_remover(Chem.MolFromSmiles(reactant))
            product = self.mc_sampling.product(self.molecule, reactant)

        mol = product
        smiles = Chem.MolToSmiles(product)
        molecule = BlockMoleculeData_wrapper(mol, smiles)
        try:
            graph = mol_to_graph(smiles)
            graph.x = torch.cat([graph.x, torch.zeros([graph.x.shape[0], 2])], dim=1)
            graph.jbond_atmidx = torch.zeros([0, 2])
            graph.jbond_preds = torch.zeros([0, 0])
            graph.stem_atmidx = torch.zeros([0])
            graph.stem_preds = torch.zeros([0, 0])
            molecule.graph = graph

            self.num_step += 1
            self.obs = {"step": self.num_step}

            reward, log_vals = self.reward(molecule, True, True, None) # always evaluate
            if self.num_step >= self.max_steps:
                env_stop = True
            else: env_stop = False
            info = {"molecule": smiles, "log_vals": log_vals}
            done = env_stop
            print(reward, info)
        except Exception as e:
            print(e)
            self.reset() # restart environment
            reward = 0.0
            log_vals = {}
            info = {"molecule": smiles, "log_vals": log_vals}
            done = False
        return self.obs, reward, done, info

        # alternatively
        # reactant = random.choices(self.mc_sampling.mols, k=1)
        # reactant = self.mc_sampling.salt_remover(Chem.MolFromSmiles(reactant))
        # products = self.mc_sampling.product(self.molecule, reactant)
        # if products is None:
        #     agent_stop = False # per current proxy_reward, when agent stop is False, we don't evaluate it
        #     env_stop = False
        #     molecule, smiles = None, None
        # else:
        #     agent_stop = True
        #     env_stop = True
        #
        #
        # self.num_step += 1
        # reward, log_vals = self.reward(molecule, agent_stop, env_stop, None)
        #
        # info = {"molecule": smiles, "log_vals": log_vals}
        # done = any((agent_stop, env_stop))
        # print (reward)
        # return None, reward, done, info

    def set_state(self, state):
        self.num_step = state
        self.obs = {"step": self.num_step}
        return self.obs

    def get_state(self):
        return self.num_step


class HTP_DummyPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)

    def compute_actions(self,
                        obs_batch=None,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        return np.array([0]), [], {}

    def learn_on_batch(self, samples):
        return {}  # return stats

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


HTP_BrutalForceTrainer = build_trainer(
    name="HTP_DummyPolicy",
    default_policy=HTP_DummyPolicy)




