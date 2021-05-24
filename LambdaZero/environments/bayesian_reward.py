import torch as th
from torch_geometric.data import Batch, DataLoader
import numpy as np
import copy
import ray
import time
import os.path as osp
import os
import threading
from rdkit import Chem
from rdkit.Chem import QED, AllChem
from rdkit import DataStructs
from torch.utils.data import Subset, ConcatDataset
import torch_geometric.transforms as T
import LambdaZero.utils
import LambdaZero.models
import LambdaZero.chem
from LambdaZero.utils import Complete, get_external_dirs
import logging
import pickle
import gzip
import pandas as pd
from LambdaZero.examples.bayesian_models.bayes_tune.deep_ensemble import DeepEnsemble
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


class PredDockBayesianReward_v1:
    def __init__(self, qed_cutoff, synth_cutoff, synth_config, binding_model, kappa,
                 soft_stop, exp, delta, simulation_cost, reward_learner,
                 regressor_config, regressor, sync_freq, dense_rewards,
                 device, transform=T.Compose([LambdaZero.utils.Complete()])):

        self.qed_cutoff = qed_cutoff
        self.soft_stop = soft_stop
        self.exp = exp
        self.delta = delta
        self.simulation_cost = simulation_cost
        self.device = device
        self.transform = transform
        self.kappa = kappa

        self.synth_cutoff = synth_cutoff
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

        self.net = LambdaZero.models.MPNNet()
        self.net.to(device)
        self.net.load_state_dict(th.load(binding_model, map_location=th.device(device)))
        self.net.eval()
        # if isinstance(regressor, MCDrop):
        self.regressor = regressor(regressor_config)
        # self.regressor.train_loader = ray.get(reward_learner.get_dataset.remote())
        self.regressor.set_weights(ray.get(reward_learner.get_weights.remote()))
        self.reward_learner_logs = ray.get(reward_learner.get_logs.remote())
        self.reward_learner = reward_learner

        self.episodes = 0
        self.aq_batches = 0
        self.weight_sync_freq = sync_freq
        self.reward_learner_logs = None
        self.train_len = 1000
        self.previous_reward = 0
        self.dense_rewards = dense_rewards

    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward
        self.episodes += 1

        if self.episodes % self.weight_sync_freq == 0:
            updates, aq_batches = ray.get(self.reward_learner.sync_weights.remote(self.aq_batches))
            if updates:
                # self.regressor.set_weights(updates['weights'])
                self.regressor.set_weights(updates['weights'])
                self.reward_learner_logs = updates['logs']
                self.train_len = updates['train_len']
                # self.regressor.train_loader = ray.get(self.reward_learner.get_dataset.remote())
                self.aq_batches = aq_batches
                print('synced weights')

    def _get_dockscore(self, molecule):
        # this is a UCB reward, but not dockscore
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        loader = DataLoader([graph])
        mean, var = self.regressor.get_mean_variance(loader, self.train_len)
        # import pdb; pdb.set_trace();
        return (mean + (self.kappa * var)).item(), {'proxy_reward_mean': mean, 'proxy_reward_var': var}

    # def _simulation(self, mol):
    #     atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
    #     graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
    #     graph = self.transform(graph)
    #     batch = Batch.from_data_list([graph]).to(self.device)
    #     pred = self.net(batch)
    #     reward = -float(pred.detach().cpu().numpy())
    #     return reward

    def _discount(self, molecule):
        mol = molecule.mol
        # QED constraint
        qed = QED.qed(mol)
        qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
        qed_discount = min(max(0.0, qed_discount), 1.0)  # relu to maxout at 1

        # Synthesizability constraint
        synth = self.synth_net(mol=mol)
        synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
        synth_discount = min(max(0.0, synth_discount), 1.0)  # relu to maxout at 1

        # Binding energy prediction
        # dockscore, log_vals = self._get_dockscore(molecule)

        dockscore_reward, log_vals = self._get_dockscore(molecule)

        #        dockscore = self._simulation(mol=mol)
        # dockscore = (self.dockscore_std[0] - dockscore) / (self.dockscore_std[1])  # normalize against std dev
        if self.reward_learner_logs is not None:
            print(self.reward_learner_logs)
            log_vals = {**log_vals, **self.reward_learner_logs}
            self.reward_learner_logs = None
        # combine rewards
        disc_reward = dockscore_reward * qed_discount * synth_discount

        self.reward_learner.add_molecule.remote(molecule, disc_reward, synth_discount * qed_discount)

        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, {"dockscore_reward": dockscore_reward,  # "chemprop_reward": dockscore,
                             "qed": qed, "synth": synth, **log_vals}

    def __call__(self, molecule, simulate, env_stop, num_steps):
        if self.dense_rewards:
            if (molecule.mol is not None) and (len(molecule.jbonds) > 0):
                # self.reward_learner.add_molecule.remote(molecule)
                # reward, log_vals = ray.get(self.reward_learner.get_reward.remote(molecule))
                # reward = reward.item()
                # import pdb;pdb.set_trace();
                reward, log_vals = self._discount(molecule)
                # pca = LambdaZero.utils.molecule_pca(molecule.mol)
                # log_vals = {}
            else:
                reward, log_vals = 0.0, {}
        else:
            if self.soft_stop:
                simulate = simulate or env_stop
            else:
                simulate = simulate

            if simulate:
                if (molecule.mol is not None) and (len(molecule.jbonds) > 0):
                    # self.reward_learner.add_molecule.remote(molecule)
                    # reward, log_vals = ray.get(self.reward_learner.get_reward.remote(molecule))
                    # reward = reward.item()
                    # import pdb;pdb.set_trace();
                    reward, log_vals = self._discount(molecule)
                    # pca = LambdaZero.utils.molecule_pca(molecule.mol)
                    # log_vals = {}
                else:
                    reward, log_vals = 0.0, {}
            else:
                reward, log_vals = 0.0, {}

        return reward, log_vals


@ray.remote
class _SimDockLet:
    def __init__(self, tmp_dir, programs_dir, datasets_dir, attribute):
        # self.dock = LambdaZero.chem.Dock_smi(tmp_dir,
        #                           osp.join(programs_dir, 'chimera'),
        #                           osp.join(programs_dir, 'dock6'),
        #                           osp.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup"),
        #                           gas_charge=True)
        print(programs_dir, "mgltools_x86_64Linux2_1.5.6")
        self.dock = LambdaZero.chem.DockVina_smi(tmp_dir,
                                                 osp.join(programs_dir, "mgltools_x86_64Linux2_1.5.6"),
                                                 osp.join(programs_dir, "vina"),
                                                 osp.join(datasets_dir, "seh/4jnc"))
        self.target_norm = [-8.6, 1.10]
        self.attribute = attribute

    def eval(self, mol):
        s = Chem.MolToSmiles(mol[1].mol)
        print("starting", s)
        try:
            r = self.dock.dock(s)[1]
        except Exception as e:  # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        setattr(mol[0], 'dockscore', th.FloatTensor([max(0, -(r - self.target_norm[0]) / self.target_norm[1])]))
        reward = -(r - self.target_norm[0]) / self.target_norm[1]
        print("done", s, r)
        return mol, max(0, reward)


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
        self.unselected_indices = []

        self.train_len = 0
        self.prune_factor = 0.2

        self.qed_cutoff = config['qed_cutoff']
        self.synth_cutoff = config['synth_config']["synth_cutoff"]
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(config['synth_config'])

        if use_dock_sim:
            self.actors = [
                _SimDockLet.remote(self.docking_loc, programs_dir, datasets_dir, self.config["data"]["target"])
                for i in range(self.num_threads)]
            self.pool = ray.util.ActorPool(self.actors)
        else:
            self.net = LambdaZero.models.MPNNet()
            self.net.to(config['device'])
            self.net.load_state_dict(th.load(binding_model, map_location=th.device(config['device'])))
            self.net.eval()
        self.target_norm = config["data"]["target_norm"]  # [-8.6, 1.10]

        # print('BR: Loaded Oracle Network')
        # print('BR: Loading Dataset ...')

        # self.dataset = config["data"]["dataset"](**config["data"]["dataset_config"])
        # ul_idxs, val_idxs, test_idxs = np.load(self.config["data"]["dataset_split_path"], allow_pickle=True)
        # print('BR: Loaded Dataset ...')

        print('BR: Preparing Dataset ...')
        # np.random.shuffle(ul_idxs) # randomly acquire batch zero
        # train_idxs = ul_idxs[:self.config["aq_size0"]]
        # self.aq_indices.extend(train_idxs)
        # ul_idxs = ul_idxs[self.config["aq_size0"]:]
        # self.unselected_indices.extend(ul_idxs)
        # train_set = Subset(self.dataset, train_idxs.tolist())
        # val_set = Subset(self.dataset, val_idxs.tolist())
        self.train_len += self.config["aq_size0"]
        # train_molecules = []
        # for i, data in enumerate(train_set):
        #     setattr(data, config["data"]["target"], -(getattr(data, config["data"]["target"]) - self.target_norm[0]) / self.target_norm[1])
        #     # import pdb; pdb.set_trace();
        #     # mol = Chem.MolFromSmiles(getattr(data, "smi"))
        #     train_molecules.append(data)
        #     # train_molecules.append([data,
        #     #     mol,
        #     #     getattr(data, config["data"]["target"]) * self.get_discount(mol)
        #     # ])
        # # import pdb;pdb.set_trace()
        # train_loader = DataLoader(train_molecules, shuffle=True, batch_size=config["data"]["b_size"])
        # self.val_loader = DataLoader(val_set, batch_size=config["data"]["b_size"])
        # self.train_molecules = train_molecules
        # self.dataset = None
        self.train_molecules, train_loader, self.val_loader = \
            self.construct_dataset(os.path.join(config["data"]["dataset_config"]["root"],
                                                config["data"]["dataset_config"]["file_names"]))
        print('BR: Prepared Dataset', "train:", len(train_loader), "val:", len(self.val_loader))

        self.regressor = config["regressor"](self.regressor_config)

        if pretrained_model is not None:
            self.regressor.model.load_state_dict(th.load(pretrained_model))
            self.regressor.model = self.regressor.model.to(device)
            print('Loaded pretrained model')
        else:
            print('BR: Running single pretraining step ...')
            scores, val = self.regressor.fit(train_loader, self.val_loader)
        # import pdb; pdb.set_trace()
        print(scores, val)
        self.logs = {**scores, **val}
        print('BR: Loaded Bayesian Reward Actor')
        self.batches = 0

    def construct_dataset(self, path):
        df = pd.read_feather(path)
        # train = df.sample(n=self.config['aq_size0'])
        # train = df.nsmallest(self.config['aq_size0'], 'dockscore')
        # val = df.sample(n=self.config['aq_size0'])
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
                    th.FloatTensor([-(row['dockscore'] - self.target_norm[0]) / self.target_norm[1]]))
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

    # def get_discount(self, mol):
    #     qed = QED.qed(mol)
    #     qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
    #     qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1
    #
    #     # Synthesizability constraint
    #     synth = self.synth_net(mol=mol)
    #     synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
    #     synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1
    #
    #     return synth_discount * qed_discount

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

    def _simulation(self, mol):
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        batch = Batch.from_data_list([graph]).to(self.device)
        pred = self.net(batch)
        reward = -float(pred.detach().cpu().numpy())
        return reward

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
        self.mols += 1
        if self.mols % self.config['num_mol_retrain'] == 0:
            print('Collected {} mols, performing batch acquisition'.format(self.mols))
            self.train_unseen_mols = self.unseen_molecules
            self.unseen_molecules = []
            self.train()

    def acquire_batch(self):
        # todo: I think we could acquire highest rewards ----------- ???
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

    def compute_targets(self, mol_set):
        if self.use_dock_sim:
            n = len(mol_set)
            t0 = time.time()
            result = np.array(list(self.pool.map(lambda a, m: a.eval.remote(m), mol_set)))
            mol_set = np.array([mol for mol in result[:, 0]]).tolist()
            rs = np.array([mol for mol in result[:, 1]])
            t1 = time.time()
            # import pdb; pdb.set_trace();
            print('Docking sim done in {}'.format(t1 - t0))
            print('Mean R in batch: ', np.mean(rs))
        else:
            n = len(mol_set)
            rs = [None] * n
            for i, (graph, mol, _, _) in enumerate(mol_set):
                r = self._simulation(mol.mol)
                setattr(graph, self.config["data"]["target"], th.FloatTensor([r]))
                rs[i] = r
                print(r)
        return mol_set, rs

    def update_with_seen(self, idxs):
        aq_mask = np.zeros(len(self.train_unseen_mols), dtype=np.bool)
        aq_mask[idxs] = True
        # import pdb; pdb.set_trace();
        aq_idxs = np.asarray(self.train_unseen_mols)[aq_mask].tolist()
        self.train_unseen_mols = np.asarray(self.train_unseen_mols)[~aq_mask].tolist()

        aq_idx, rews = self.compute_targets(aq_idxs)
        rews = np.array(rews)

        if len(self.unselected_indices) == 0:
            self.unselected_indices = self.aq_indices
            self.aq_indices = []

        # aq_idxs = np.random.choice(self.unselected_indices, self.config["aq_size"] // 8, replace=False)
        # self.aq_indices.extend(aq_idxs)
        # aq_mask = np.zeros(len(self.dataset), dtype=np.bool)
        # aq_mask[self.aq_indices] = True
        # self.unselected_indices = np.arange(len(self.dataset))[~aq_mask].tolist()
        # aq_set = self.dataset[aq_idxs.tolist()]

        loader = DataLoader(np.array(aq_idx)[:, 0], shuffle=False, batch_size=self.config["data"]["b_size"])
        mean, var = self.regressor.get_mean_variance(loader, self.train_len)
        mse_before = np.mean((rews - mean) ** 2)

        # aq_set_arr = np.array([graph for graph in aq_set])
        # [graph,
        # Chem.MolFromSmiles(getattr(graph, "smi")),
        # getattr(graph, self.config["data"]["target"]) * self.get_discount(Chem.MolFromSmiles(getattr(graph, "smi")))]
        # for graph in aq_set])
        # fine_tune_dataset = np.concatenate((np.array(aq_idx)[:, :1], aq_set_arr), axis=0)
        # import pdb; pdb.set_trace()
        self.train_molecules.extend(np.array(aq_idx)[:, :1].reshape(-1).tolist())

        print("self train_molecules", self.train_molecules)
        train_loader = DataLoader(self.train_molecules, shuffle=True, batch_size=self.config["data"]["b_size"])
        # import pdb; pdb.set_trace();
        self.unseen_molecules.extend(self.train_unseen_mols)

        # fit model to the data
        if self.batches % 5 == 0:
            scores, val = self.regressor.fit(train_loader, self.val_loader, False)
            # scores, val = self.regressor.fine_tune(fine_tune_loader, self.val_loader, False)
        else:
            scores, val = self.regressor.fit(train_loader, self.val_loader, False)
            # scores, val = self.regressor.fine_tune(fine_tune_loader, self.val_loader, False)

        self.train_len += self.config["aq_size"]
        mean, var = self.regressor.get_mean_variance(loader, self.train_len)
        mse_after = np.mean((rews - mean) ** 2)
        smiles = []
        for i in range(len(aq_idx)):
            smiles.append([Chem.MolToSmiles(aq_idx[i][1].mol), rews[i]])
            rews[i] = rews[i] * aq_idx[i][3]
        pickle.dump(smiles, gzip.open(os.path.join(self.mol_dump_loc, f'batch-{self.batches + 1}.pkl.gz'), 'wb'))
        self.regressor.save(self.mol_dump_loc)
        self.aq_molecules.extend(np.array(aq_idx)[:, :3].tolist())

        self.logs = {**scores, **val, 'mse_before': mse_before, 'mse_after': mse_after,
                     'mse_diff': mse_before - mse_after,
                     'oracle_max': np.max(rews), 'oracle_mean': np.mean(rews)}
        print(self.logs)
        return scores

    def train(self):
        print('BR: Acquring Batch ...')
        idx = self.acquire_batch()
        print("acquired idx:", idx)
        print('BR: Acquired Batch')
        print('BR: Retraining with Acquired Batch ...')
        self.update_with_seen(idx)
        print('Bayesian Reward Actor: Retrained')
        self.batches += 1
