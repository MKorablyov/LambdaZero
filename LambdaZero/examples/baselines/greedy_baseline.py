from LambdaZero.environments import chemMDP, MolMDP, reward, BlockMolEnv_v4
from LambdaZero.environments.block_mol_v4 import DEFAULT_CONFIG as env_config
import LambdaZero
import time
import os.path as osp
import random
import pandas as pd
import time
import numpy as np
import ray
from rdkit import Chem
from copy import copy, deepcopy
#import concurrent.futures
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

default_config = {
    "env": BlockMolEnv_v4,
    "env_config": env_config
}

def step_env(env, state, action_mask):
    actions = np.where(action_mask)[0]
    #molecules = []
    next_states = []
    observations = []
    values = []

    for i, a in enumerate(actions):
        env.set_state(state)
        obs, reward, _, info = env.step(a)
        #molecules.append(deepcopy(info["molecule"].mol))
        next_states.append(env.get_state())
        observations.append(obs)
        values.append(reward)
        #print('1')

    #values = evaluate_molecules(molecules)
    value = max(values)
    next_state = next_states[values.index(max(values))]
    observation = observations[values.index(max(values))]
    return value, next_state, observation

#@ray.remote
def greedy_search(env, state, obs, max_steps, stop_condition=None, top_k=1):
    start = time.time()
    values = []
    #states = []
    for i in range(max_steps):
        action_mask = obs["action_mask"]
        val, state, obs = step_env(env, state, action_mask)

        #print(("Finished iteration {}, current max: {:.3f}").format(i, val))
        values.append(val)
        #states.append(state)
        #
        # print((
        #     "Finished iteration {}, current max: {:.3f}, total evals: {}"
        # ).format(i, np.max(values), total_evals()))

        if stop_condition is not None:
            stop_condition.update(val, state, obs)
            if stop_condition.should_stop():
                break

    end = time.time()
    times = end-start
    #print("time", end - start)
    return max(values), times
    # top_idx = np.argsort(values)[-top_k:]
    # return tuple(zip(*[(values[i], states[i]) for i in top_idx]))

if __name__ == "__main__":

    ray.init()
    # dockactors = [DockActor.remote() for _ in range(multiprocessing.cpu_count())]
    # pool = util.ActorPool(dockactors)
    reward_func = reward.PredDockReward_v2(default_config['env_config']["reward_config"]['binding_model'],
                                           default_config['env_config']["reward_config"]['qed_cutoff'],
                                           default_config['env_config']["reward_config"]['synth_cutoff'],
                                           default_config['env_config']["reward_config"]['synth_config'],
                                           default_config['env_config']["reward_config"]['soft_stop'],
                                           default_config['env_config']["reward_config"]['exp'], default_config['env_config']["reward_config"]['delta'],
                                           default_config['env_config']["reward_config"]['simulation_cost'],
                                           default_config['env_config']["reward_config"]['device'])

    # def evaluate_molecules(molecules):
    #     rewards = []
    #     for molecule in molecules:
    #         #print (type(molecule))
    #         dock = reward_func._simulation(molecule)
    #         if dock is not None:
    #             discounted_reward, log_val = reward_func._discount(molecule, dock)
    #             rewards.append(discounted_reward)
    #     return rewards
    #     #return list(
    #     #    pool.map(lambda a, m: a.evaluate_molecule.remote(m), molecules))

    config = default_config
    config["env_config"]["reward_config"]["device"] = "cpu"
    env = config["env"](config["env_config"])
    times = 0
    for i in range(10000):

        obs = env.reset()
        state = env.get_state()
        # values, states = greedy_search(env, state, obs, 8, None, 1)
        values, timesz = greedy_search(env, state, obs, 8, None, 1)
        times += timesz
        #print(states)
        print("highest value is", values)
    print (times/10000)
#
#
# config = {
#     "binding_model": osp.join(datasets_dir, "brutal_dock/mpro_6lze/trained_weights/vanilla_mpnn/model.pth"),
#     "qed_cutoff": [0.2, 0.7],
#     "synth_cutoff": [0, 4],
#     "soft_stop": True,
#     "exp": None,
#     "delta": False,
#     "simulation_cost": 0.0,
#     "device": "cpu",
#     "synth_config": {
#         "trainer_config": {
#             "dataset_type": "regression",
#             "train_dataset": None,
#             "save_dir": None,
#
#             "features_generator": None,  # Method(s) of generating additional features
#             "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
#             "no_features_scaling": False,  # Turn off scaling of features
#             "num_workers": 8,  # Number of workers for the parallel data loading (0 means sequential)
#             "batch_size": 50,  # Batch size
#             "separate_val_path": None,  # Path to separate val set, optional
#             "separate_test_path": None,  # Path to separate test set, optional
#             "split_type": "random",
#             "split_sizes": (0.8, 0.1, 0.1),
#             "num_folds": 1,
#             "seed": 0,
#             "pytorch_seed": 0,
#             "log_frequency": 10,
#             "cache_cutoff": 10000,
#             "save_smiles_splits": False,
#
#             "hidden_size": 300,
#             "depth": 3,
#             "dropout": 0.0,
#             "activation": "ReLu",
#             "ffn_num_layers": 2,
#             "ensemble_size": 1,
#             "atom_messages": False,  # Centers messages on atoms instead of on bonds
#             "undirected": False,
#
#             "epochs": 150,
#             "warmup_epochs": 2.0,  # epochs for which lr increases linearly; afterwards decreases exponentially
#             "init_lr": 1e-4,  # Initial learning rate
#             "max_lr": 1e-3,  # Maximum learning rate
#             "final_lr":  1e-4,  # Final learning rate
#             "class_balance": False,
#             },
#         "predict_config": {
#             "dataset_type": "regression",
#             "features_generator": None,
#             "features_path": None,  # Path(s) to features to use in FNN (instead of features_generator)
#             "no_features_scaling": False,  # Turn off scaling of features
#             "num_workers": 8,  # Number of workers for the parallel data loading (0 means sequential)
#             "batch_size": 50,  # Batch size
#             "disable_progress_bar": True,
#             "checkpoint_path": osp.join(datasets_dir, "Synthesizability/MPNN_model/Regression/model_1/model.pt")
#         },
#     }
# }
#
# smiles = []
# starttime = time.clock()
# datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
#
# ChemMDP = chemMDP()
# reward_func = reward.PredDockReward_v2(config['binding_model'], config['qed_cutoff'], config['synth_cutoff'], config['synth_config'],
#                  config['soft_stop'], config['exp'], config['delta'], config['simulation_cost'], config['device'])
# # for j in range(90):
# reward = []
# log_vals = []
# smiles = []
# i = 0
# #     for i in range(10000):
# # greedy
# while i < 20000:
#     random_steps = random.randint(10, 30)
#     ChemMDP.reset()
#     ChemMDP.random_walk(length=5)
#     prev_reward = 0
#     j = 0
#     print('random_step is', random_steps)
#     fail = 0
#     while j < random_steps:
#         prev_molecule = copy(ChemMDP.molecule)
#         ChemMDP.random_walk(length=j+1, step=j)
#         dock = reward_func._simulation(ChemMDP.molecule.GetMol())
#
#         if dock is not None:
#             discounted_reward, log_val = reward_func._discount(ChemMDP.molecule.GetMol(), dock)
#             if discounted_reward > prev_reward:
#                 #print ('yes')
#                 j += 1
#                 print (discounted_reward)
#                 prev_reward = copy(discounted_reward)
#             else:
#                 #print('oui')
#                 fail += 1
#                 if fail >= 2550:
#                     break
#                 ChemMDP.molecule = copy(prev_molecule)
#             # if log_val['synth'] > 3.5:
#             #     #print('yes')
#             #     reward.append(discounted_reward)
#             #     log_vals.append([log_val])
#             #     smiles.append(Chem.MolToSmiles(ChemMDP.molecule))
#             #     i += 1
#                 # graph = Chem.Draw.MolToFile(ChemMDP.molecule,
#                 #                            '/Users/chenghaoliu/ML/Experiments/LambdaZero/ChemMDP/random_walk/synth/1/' + str(
#                 #                                i) + '.png')
#             # print ('y')
#         else:
#             discounted_reward, log_val = -0.5, {"dock_reward": -0.5, "natm": 0.0, "qed": -0.5, "synth": -0.5}
#             fail += 1
#             if fail >= 150:
#                 break
#     print ('done')
#     i += 1
#
#     #ChemMDP.random_walk(random_steps)
#
# # while i < 20000:
# #     random_steps = random.randint(10, 30)
# #     ChemMDP.reset()
# #     ChemMDP.random_walk(random_steps)
# #     dock = reward_func._simulation(ChemMDP.molecule.GetMol())
# #     if dock is not None:
# #         discounted_reward, log_val = reward_func._discount(ChemMDP.molecule.GetMol(), dock)
# #         if log_val['synth'] > 3.5:
# #             print('yes')
# #             reward.append(discounted_reward)
# #             log_vals.append([log_val])
# #             smiles.append(Chem.MolToSmiles(ChemMDP.molecule))
# #             i += 1
# #             #graph = Chem.Draw.MolToFile(ChemMDP.molecule,
# #             #                            '/Users/chenghaoliu/ML/Experiments/LambdaZero/ChemMDP/random_walk/synth/1/' + str(
# #             #                                i) + '.png')
# #         #print ('y')
# #     else:
# #         discounted_reward, log_val = -0.5, {"dock_reward": -0.5, "natm": 0.0, "qed": -0.5, "synth": -0.5}
# #     #discounted_reward, log_val = reward_func._discount(ChemMDP.molecule, reward_func._simulation(ChemMDP.molecule))
# #
# #     #smile = Chem.MolToSmiles(ChemMDP.molecule)
# #     #smiles.append(smile)
# #     #graph = Chem.Draw.MolToFile(ChemMDP.molecule, '/Users/chenghaoliu/ML/Experiments/LambdaZero/ChemMDP/random_walk/10-50/'+str(i)+'.png' )
# df = pd.DataFrame(reward)
# df_1 = pd.DataFrame(log_vals)
# df_2 = pd.DataFrame(smiles)
#
# df.to_csv('/Users/chenghaoliu/ML/Experiments/LambdaZero/ChemMDP/random_walk/synth/0_submit/discounted_reward__greedy_'+str(i)+'.csv', header=False, index=False)
# df_1.to_csv('/Users/chenghaoliu/ML/Experiments/LambdaZero/ChemMDP/random_walk/synth/0_submit/log_vals__greedy_'+str(i)+'.csv', header=False, index=False)
# df_2.to_csv('/Users/chenghaoliu/ML/Experiments/LambdaZero/ChemMDP/random_walk/synth/0_submit/molecule__greedy_'+str(i)+'.csv', header=False, index=False)