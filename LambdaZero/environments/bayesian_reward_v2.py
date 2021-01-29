import pickle, gzip,time,  os.path as osp
import torch
import pandas as pd
import numpy as np
import ray

from torch_geometric.data import Batch, DataLoader
import torch_geometric.transforms as T
from rdkit import Chem
from rdkit.Chem import QED

import LambdaZero.utils
import LambdaZero.models
import LambdaZero.chem

raise ImportError
# todo: I want to be able to:
#  a) share bottom weights including with agent
#  b) have any model of uncertainty such as BRR/evidential/etc/docking and any acquisition function
#  c) I want to be able to run the same acquisition function on a static dataset
#  d*) I want to run BRR in online (just choose a small update interval should be fine)


# Reward
# __init__(CombinedScoreInstance)
#   self.reward_actor = CombinedScoreInstance.get_actor()
#
# __call__(molecule)
#   combined_score_instance.add_molecule()
#   return self.reward_actor(molecule)


# CombinedScoreProxy@remote
# __init__(DockScoreUCB, SynthScore)
# add_molecule()
#   if 1000 molecules:
#       idx = self.acquire_and_update()
#
# def acquire_and_update():
#   logs = self.DockScore.acquire_and_update(self.molecules, self.scores, self.discounts)
#   return logs
#
# def get_actor(): # this will make a local actor to compute scores
#   return rewardActor


# DockScoreUCB(UCB):
#
# acquire():
#   do_docking
#
# acquire_and_update(molecules, scores, discounts)
#    molecules, aq_vals, discounts = self.acquire_batch()
#    self.dock(molecules)
#    self.update_with_seen(molecules, dockscores, discounts)


# UCB:
# __init__(model, dataset)
#
# update_with_seen(molecules, scores, discounts)
#   train network on data
#
# get_scores()
#
# acquire_batch(molecules, scores, discounts)
#

# ProxyScoreActor
# __init__(get_weights_func):
# __call__(molecule):
#   if_update_weights:
    # welf.weights = self.get_weights_func.remote.get_weights()
    # return score







# fixme: this might not be the best way to construct the dataset
def load_dataset(path):
    df = pd.read_feather(path)
    train, val, _ = np.split(df.sample(n=2 * self.config['aq_size0']),
                             [self.config['aq_size0'], 2 * self.config['aq_size0']])  # splits without replacement
    train_mols = []
    # import pdb; pdb.set_trace()
    for index, row in train.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        setattr(graph, self.config["data"]["target"],
                torch.FloatTensor([-(row['dockscore'] - self.target_norm[0]) / self.target_norm[1]]))
        train_mols.append(graph)
    train_loader = DataLoader(train_mols, shuffle=True, batch_size=self.config["data"]["b_size"])

    val_mols = []
    for i, row in val.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        setattr(graph, self.config["data"]["target"],
                -(row['dockscore'] - self.target_norm[0]) / self.target_norm[1])
        val_mols.append(graph)
    val_loader = DataLoader(val_mols, shuffle=True, batch_size=self.config["data"]["b_size"])
    return train_mols, train_loader, val_loader


@ray.remote
class _SimDockLet:
    def __init__(self, tmp_dir, programs_dir, datasets_dir, attribute):
        print(programs_dir, "mgltools_x86_64Linux2_1.5.6")
        self.dock = LambdaZero.chem.DockVina_smi(tmp_dir,
                                                 osp.join(programs_dir, "mgltools_x86_64Linux2_1.5.6"),
                                                 osp.join(programs_dir, "vina"),
                                                 osp.join(datasets_dir, "seh/4jnc"))

        self.target_norm = [-8.6, 1.10] # fixme ?
        self.attribute = attribute

    def eval(self, mol):
        s = Chem.MolToSmiles(mol[1].mol)
        print("starting", s)
        try:
            _, r, _ = self.dock.dock(s)
        except Exception as e:  # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        setattr(mol[0], 'dockscore', torch.FloatTensor([max(0, -(r - self.target_norm[0]) / self.target_norm[1])]))
        reward = -(r - self.target_norm[0]) / self.target_norm[1]
        print("done", s, r)
        return mol, max(0, reward)


def dock_molecules(mol_set):  # this will compute docking energies
    n = len(mol_set)
    t0 = time.time()
    result = np.array(list(self.pool.map(lambda a, m: a.eval.remote(m), mol_set)))
    mol_set = np.array([mol for mol in result[:, 0]]).tolist()
    rs = np.array([mol for mol in result[:, 1]])
    t1 = time.time()

    print('Docking sim done in {}'.format(t1 - t0))
    print('Mean R in batch: ', np.mean(rs))
    return mol_set, rs



class DockScoreUCB:
    def __init__(self, config):
        self.transform = config["transform"]
        self.kappa = config["kappa"]
        self.weight_sync_freq = config["weight_sync_freq"]

        self.learner = config["reward_learner"]  # todo: some naming system for these
        self.regressor = config["regressor"](config["regressor_config"])
        self.regressor.set_weights(ray.get(self.learner.get_weights.remote()))

        self.num_calls = 0
        self.train_len = None # would be set up with call 0

    def _sync_learner(self):
        updates, aq_batches = ray.get(self.learner.sync_weights.remote(self.aq_batches))
        if updates:
            self.regressor.set_weights(updates['weights'])
            self.reward_learner_logs = updates['logs']
            self.train_len = updates['train_len']
            self.aq_batches = aq_batches
            print('synced weights')

    def __call__(self, molecule):
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        loader = DataLoader([graph])
        mean, var = self.regressor.get_mean_variance(loader, self.train_len)

        if self.num_calls % self.weight_sync_freq == 0: self._sync_learner()
        self.num_calls += 1
        return (mean + (self.kappa * var)).item(), {'proxy_reward_mean': mean, 'proxy_reward_var': var}

class SynthScore:
    def __init__(self, config):
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(config)

    def __call__(self, molecule):
        return self.synth_net(mol=molecule.mol), {}


class CombinedScore:
    def __init__(self, config):
        self.qed_cutoff = config["qed_cutoff"]
        self.synth_cutoff = config["synth_cutoff"]

        self.dockScore = DockScoreUCB(config)
        self.synthScore = SynthScore(config)

    def __call__(self, molecule):
        # QED constraint
        qed_score = QED.qed(molecule.mol)
        qed_discount = (qed_score - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
        qed_discount = min(max(0.0, qed_discount), 1.0)  # relu to maxout at 1

        # Synthesizability constraint
        synth_score, log_synth = self.synthScore(molecule)
        synth_discount = (synth_score - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
        synth_discount = min(max(0.0, synth_discount), 1.0)  # relu to maxout at 1

        # Binding energy prediction
        dock_score, log_dock = self.dockScore(molecule)

        # fixme - need to build a logger for the acquirer ??
        #if self.reward_learner_logs is not None:
        #    print(self.reward_learner_logs)
        #    log_vals = {**log_vals, **self.reward_learner_logs}
        #    self.reward_learner_logs = None

        # combine rewards
        comb_score = dock_score * qed_discount * synth_discount
        return comb_score, {"qed_score": qed_score, "synth_score": synth_score, "dock_score":dock_score,
                            "comb_score": comb_score, **log_synth, **log_dock}


def _discount(molecule):
    # we need to acquire molecules that increase total reward ?
    self.reward_learner.add_molecule.remote(molecule, disc_reward, synth_discount * qed_discount)
    # # fixme - it would be more intuitive to compute this with handling exceptions
    # if self.exp is not None: disc_reward = self.exp ** disc_reward

    # # delta reward
    # delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
    # self.previous_reward = disc_reward
    # if self.delta: disc_reward = delta_reward
    # return disc_reward,



datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

class PredDockBayesianReward_v1:
    def __init__(self, qed_cutoff, synth_cutoff, synth_config, binding_model, kappa,
                 soft_stop, exp, delta, simulation_cost, reward_learner,
                 regressor_config, regressor, sync_freq, dense_rewards,
                 device):

        self.qed_cutoff = qed_cutoff
        self.soft_stop = soft_stop
        self.exp = exp # todo: if exp if delta
        self.delta = delta
        self.simulation_cost = simulation_cost
        #self.device = device

        #self.transform = transform
        #self.kappa = kappa
        #self.synth_cutoff = synth_cutoff

        # fixme - I am not sure what logs would be for here ??
        self.reward_learner_logs = ray.get(reward_learner.get_logs.remote())
        self.reward_learner = reward_learner

        self.episodes = 0
        self.aq_batches = 0
        self.weight_sync_freq = sync_freq
        self.reward_learner_logs = None

        # fixme ! - this does not seem right
        self.train_len = 1000
        self.previous_reward = 0
        self.dense_rewards = dense_rewards

    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward

    def __call__(self, molecule, simulate, env_stop, num_steps):
        if self.dense_rewards:
            if (molecule.mol is not None) and (len(molecule.jbonds) > 0):
                reward, log_vals = 0.0, {}
        else:
            if self.soft_stop:
                simulate = simulate or env_stop
            else:
                simulate = simulate

            if simulate:
                if (molecule.mol is not None) and (len(molecule.jbonds) > 0):
                    reward, log_vals = self._discount(molecule)
                else:
                    reward, log_vals = 0.0, {}
            else:
                reward, log_vals = 0.0, {}

        return reward, log_vals




@ray.remote(num_gpus=1, num_cpus=4)
class BayesianRewardActor():
    def __init__(self, config, use_dock_sim, binding_model,
                 pretrained_model=None, transform=T.Compose([LambdaZero.utils.Complete()])):

        self.device = config["device"]
        self.transform = transform
        self.config = config
        self.regressor_config = config["regressor_config"]
        self.mol_dump_loc = config["mol_dump_loc"]
        self.docking_loc = config["docking_loc"]
        self.unseen_molecules = []
        self.aq_molecules = []
        self.mols = 0
        self.use_dock_sim = use_dock_sim
        self.num_threads = 8
        self.max_size = 25000
        self.temperature = config.get('temperature', 2)
        self.aq_indices = []

        self.train_len = 0
        self.prune_factor = 0.2

        self.qed_cutoff = config['qed_cutoff']
        self.synth_cutoff = config['synth_config']["synth_cutoff"]
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(config['synth_config'])

        #if use_dock_sim:
        self.actors = [_SimDockLet.remote(self.docking_loc, programs_dir, datasets_dir, self.config["data"]["target"])
                       for i in range(self.num_threads)]
        self.pool = ray.util.ActorPool(self.actors)
        # else:
        #     self.net = LambdaZero.models.MPNNet()
        #     self.net.to(config['device'])
        #     self.net.load_state_dict(th.load(binding_model, map_location=torch.device(config['device'])))
        #     self.net.eval()
        # self.target_norm = config["data"]["target_norm"]  # [-8.6, 1.10]

        print('BR: Preparing Dataset ...')

        self.train_len += self.config["aq_size0"]
        self.train_molecules, train_loader, self.val_loader = \
            load_dataset(osp.join(config["data"]["dataset_config"]["root"],
                                                config["data"]["dataset_config"]["file_names"]))
        print('BR: Prepared Dataset', "train:", len(train_loader), "val:", len(self.val_loader))

        self.regressor = config["regressor"](self.regressor_config)

        if pretrained_model is not None:
            self.regressor.model.load_state_dict(torch.load(pretrained_model))
            self.regressor.model = self.regressor.model.to(device)
            print('Loaded pretrained model')
        else:
            print('BR: Running single pretraining step ...')
            scores, val = self.regressor.fit(train_loader, self.val_loader)

        #print(scores, val)
        self.logs = {**scores, **val}
        print('BR: Loaded Bayesian Reward Actor')
        self.batches = 0

    def prune(self):
        new_mols_idx = np.argsort([i[2] for i in self.unseen_molecules])[int(self.prune_factor * self.max_size):]
        if len(self.unseen_molecules) < self.max_size:
            return
        new_mols = []
        for i, j in enumerate(new_mols_idx):
            new_mols.append(self.unseen_molecules[j])
        self.unseen_molecules = new_mols
        print('pruned', len(self.unseen_molecules))

    def sample(self):
        if len(self.aq_molecules) > 0:
            idx = np.random.randint(len(self.aq_molecules))
            return self.aq_molecules[idx][1]
        else:
            return None

    def get_weights(self):
        return self.regressor.get_weights()

    def get_logs(self):
        return self.logs

    def get_dataset(self):
        return self.train_loader

    def sync_weights(self, aq_batches):
        if self.batches > aq_batches:
            return {
                       'weights': self.get_weights(),
                       'logs': self.logs,
                       'train_len': self.train_len
                   }, self.batches
        return None, None

    def add_molecule(self, molecule, disc_rew, discount):
        if len(self.unseen_molecules) > self.max_size:
            self.prune()
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)

        self.unseen_molecules.append([graph, molecule, disc_rew, discount])
        self.mols += 1 # num

        if self.mols % self.config['num_mol_retrain'] == 0:
            print('Collected {} mols, performing batch acquisition'.format(self.mols))
            self.train_unseen_mols = self.unseen_molecules
            self.unseen_molecules = []
            self.train()

    def acquire_batch(self):
        # fixme: this computes aq_value * discount; but we already have disc reward for each molecule ???
        mols = np.array(self.train_unseen_mols)[:, 0]
        loader = DataLoader(mols, batch_size=self.config["data"]["b_size"], num_workers=2, pin_memory=True)
        mean, var = self.regressor.get_mean_variance(loader, self.train_len)

        scores = mean + (self.config["kappa"] * var)
        for i in range(len(self.train_unseen_mols)):
            scores[i] = self.train_unseen_mols[i][3] * scores[i]
            self.train_unseen_mols[i][2] = self.train_unseen_mols[i][3] * scores[i]

        # noise is not a part of original UCT but is added here
        if self.config["minimize_objective"]: scores = -scores
        idxs = np.argsort(-scores)[:self.config["aq_size"]]
        return idxs

    def update_with_seen(self, idxs):
        # todo: this keep unseen molecules from past iterations ?
        aq_mask = np.zeros(len(self.train_unseen_mols), dtype=np.bool)
        aq_mask[idxs] = True
        aq_idxs = np.asarray(self.train_unseen_mols)[aq_mask].tolist()
        self.train_unseen_mols = np.asarray(self.train_unseen_mols)[~aq_mask].tolist()

        aq_idx, dockscores = dock_molecules(aq_idxs)
        dockscores = np.array(dockscores)

        # todo: this would estimate MSE
        #loader = DataLoader(np.array(aq_idx)[:, 0], shuffle=False, batch_size=self.config["data"]["b_size"])
        #mean, var = self.regressor.get_mean_variance(loader, self.train_len)
        #mse_before = np.mean((rews - mean) ** 2)

        # todo:
        #  unseen_molecules: these are proposed_molecules from actors
        #  train_unseen_molecules are same molecules during training


        self.train_molecules.extend(np.array(aq_idx)[:, :1].reshape(-1).tolist())
        print("self train_molecules", self.train_molecules)
        train_loader = DataLoader(self.train_molecules, shuffle=True, batch_size=self.config["data"]["b_size"])
        self.unseen_molecules.extend(self.train_unseen_mols)

        # fit model to the data
        #if self.batches % 5 == 0:
        # scores, val = self.regressor.fit(train_loader, self.val_loader, False)
        # #else:
        # #    scores, val = self.regressor.fit(train_loader, self.val_loader, False)
        #
        # self.train_len += self.config["aq_size"]
        # mean, var = self.regressor.get_mean_variance(loader, self.train_len)
        # mse_after = np.mean((rews - mean) ** 2)
        # smiles = []
        # for i in range(len(aq_idx)):
        #     smiles.append([Chem.MolToSmiles(aq_idx[i][1].mol), rews[i]])
        #     rews[i] = rews[i] * aq_idx[i][3]
        # pickle.dump(smiles, gzip.open(osp.join(self.mol_dump_loc, f'batch-{self.batches + 1}.pkl.gz'), 'wb'))
        # self.regressor.save(self.mol_dump_loc)

        self.aq_molecules.extend(np.array(aq_idx)[:, :3].tolist())

        self.logs = {**scores, **val, 'mse_before': mse_before, 'mse_after': mse_after,
                     'mse_diff': mse_before - mse_after,
                     'oracle_max': np.max(rews), 'oracle_mean': np.mean(rews)}
        #print(self.logs)
        return scores

    def train(self):
        print('BR: Acquring Batch ...')
        idx = self.acquire_batch()
        print("acquired idx:", idx)
        self.update_with_seen(idx)
        print('Bayesian Reward Actor: Retrained')
        self.batches += 1
