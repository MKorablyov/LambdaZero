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

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

class PredDockReward:
    def __init__(self, load_model, natm_cutoff, qed_cutoff, soft_stop, exp, delta, simulation_cost, device,
                 transform=T.Compose([LambdaZero.utils.Complete()])):

        self.natm_cutoff = natm_cutoff
        self.qed_cutoff = qed_cutoff
        self.soft_stop = soft_stop
        self.exp = exp
        self.delta = delta
        self.simulation_cost = simulation_cost
        self.device = device
        self.transform = transform

        self.net = LambdaZero.models.MPNNet()
        self.net.to(device)
        self.net.load_state_dict(th.load(load_model, map_location=th.device(device)))
        self.net.eval()

    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward

    def _discount(self, mol, reward):
        # num atoms constraint
        natm = mol.GetNumAtoms()
        natm_discount = (self.natm_cutoff[1] - natm) / (self.natm_cutoff[1] - self.natm_cutoff[0])
        natm_discount = min(max(natm_discount, 0.0), 1.0) # relu to maxout at 1

        # QED constraint
        qed = QED.qed(mol)
        qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
        qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1
        disc_reward = min(reward, reward * natm_discount * qed_discount) # don't appy to negative rewards
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, qed

    def _simulation(self, molecule):
        mol = molecule.mol
        if (mol is not None) and (len(molecule.jbonds) > 0):
            atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
            graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
            graph = self.transform(graph)
            batch = Batch.from_data_list([graph]).to(self.device)
            pred = self.net(batch)
            reward = -float(pred.detach().cpu().numpy())
        else:
            reward = None
        return reward

    def __call__(self, molecule, simulate, env_stop, num_steps):
        if self.soft_stop:
            simulate = simulate or env_stop
        else:
            simulate = simulate

        if simulate:
            reward = self._simulation(molecule)
            if reward is not None:
                discounted_reward, qed = self._discount(molecule.mol, reward)
            else:
                reward, discounted_reward, qed = -0.5, -0.5, -0.5
        else:
            reward, discounted_reward, qed = 0.0, 0.0, 0.0
        return discounted_reward, {"reward": reward, "discounted_reward": discounted_reward, "QED": qed}

class PredDockReward_v2:
    def __init__(self, binding_model, qed_cutoff, synth_cutoff, synth_config,
                 soft_stop, exp, delta, simulation_cost, device):

        #self.natm_cutoff = natm_cutoff
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.soft_stop = soft_stop
        self.exp = exp
        self.delta = delta
        self.simulation_cost = simulation_cost
        self.device = device
        self.transform = T.Compose([LambdaZero.utils.Complete()])

        self.net = LambdaZero.models.MPNNet()
        self.net.to(device)
        self.net.load_state_dict(th.load(binding_model, map_location=th.device(device)))
        self.net.eval()

        self.synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward

    def _discount(self, mol, reward):

        # num atoms constraint
        natm = mol.GetNumAtoms()
        #natm_discount = (self.natm_cutoff[1] - natm) / (self.natm_cutoff[1] - self.natm_cutoff[0])
        #natm_discount = min(max(natm_discount, 0.0), 1.0) # relu to maxout at 1

        # QED constraint
        qed = QED.qed(mol)
        qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
        qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1

        # Synthesizability constraint
        synth = self.synth_net(mol=mol)
        synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
        synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1

        #print("synth discount", synth_discount, synth )

        # combine rewards
        disc_reward = reward * qed_discount * synth_discount
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, {"dock_reward": reward, "natm": natm, "qed" : qed, "synth" : synth}

    def _simulation(self, molecule):
        mol = molecule.mol
        if (mol is not None) and (len(molecule.jbonds) > 0):
            atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
            graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
            graph = self.transform(graph)
            batch = Batch.from_data_list([graph]).to(self.device)
            pred = self.net(batch)
            reward = -float(pred.detach().cpu().numpy())
        else:
            reward = None
        return reward

    def __call__(self, molecule, simulate, env_stop, num_steps):
        if self.soft_stop:
            simulate = simulate or env_stop
        else:
            simulate = simulate

        if simulate:
            reward = self._simulation(molecule)
            if reward is not None:
                discounted_reward, log_vals = self._discount(molecule.mol, reward)
            else:
                discounted_reward, log_vals = -0.5, {"dock_reward": -0.5, "natm": 0.0, "qed" : -0.5, "synth" : -0.5}
        else:
            discounted_reward, log_vals = 0.0, {}
        return discounted_reward, log_vals


class PredDockReward_v3:
    def __init__(self, qed_cutoff, synth_config, dockscore_config,
                 soft_stop, exp, delta, simulation_cost, device, transform=T.Compose([LambdaZero.utils.Complete()])):

        self.qed_cutoff = qed_cutoff
        self.soft_stop = soft_stop
        self.exp = exp
        self.delta = delta
        self.simulation_cost = simulation_cost
        self.device = device
        self.transform = transform

        self.synth_cutoff = synth_config["synth_cutoff"]
        self.dockscore_std = dockscore_config["dockscore_std"]
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)
        self.binding_net = LambdaZero.models.ChempropWrapper_v1(dockscore_config)
    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward

    def _discount(self, mol):
        # QED constraint
        qed = QED.qed(mol)
        qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
        qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1

        # Synthesizability constraint
        synth = self.synth_net(mol=mol)
        synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
        synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1

        # Binding energy prediction
        dockscore = self.binding_net(mol=mol)
        dockscore_reward = (self.dockscore_std[0] - dockscore) / (self.dockscore_std[1])  # normalize against std dev

        # combine rewards
        disc_reward = dockscore_reward * qed_discount * synth_discount
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, {"dockscore_reward": dockscore_reward, "qed": qed, "synth": synth}

    def __call__(self, molecule, simulate, env_stop, num_steps):
        if self.soft_stop:
            simulate = simulate or env_stop
        else:
            simulate = simulate

        if simulate:
            if (molecule.mol is not None) and (len(molecule.jbonds) > 0):
                discounted_reward, log_vals = self._discount(molecule.mol)
                # pca = LambdaZero.utils.molecule_pca(molecule.mol)
                # log_vals = {**pca, **log_vals}
            else:
                discounted_reward, log_vals = 0.0, {}
        else:
            discounted_reward, log_vals = 0.0, {}

        return discounted_reward, log_vals

class PredDockBayesianReward_v1:
    def __init__(self, qed_cutoff, synth_config, binding_model, kappa,
                 soft_stop, exp, delta, simulation_cost, reward_learner, regressor_config,
                 regressor, sync_freq, 
                 device, transform=T.Compose([LambdaZero.utils.Complete()])):

        self.qed_cutoff = qed_cutoff
        self.soft_stop = soft_stop
        self.exp = exp
        self.delta = delta
        self.simulation_cost = simulation_cost
        self.device = device
        self.transform = transform
        self.kappa = kappa

        self.synth_cutoff = synth_config["synth_cutoff"]
        self.dockscore_std = [-43.042, 7.057]
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)
        
        self.net = LambdaZero.models.MPNNet()
        self.net.to(device)
        self.net.load_state_dict(th.load(binding_model, map_location=th.device(device)))
        self.net.eval()
        
        self.regressor = regressor(regressor_config)
        # self.regressor.train_loader = ray.get(reward_learner.get_dataset.remote())
        self.regressor.model.load_state_dict(ray.get(reward_learner.get_weights.remote()))
        self.reward_learner_logs = ray.get(reward_learner.get_logs.remote())
        self.reward_learner = reward_learner

        self.episodes = 0
        self.aq_batches = 0
        self.weight_sync_freq = sync_freq
        self.reward_learner_logs = None
        self.train_len = 1000

    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward
        self.episodes += 1
        if self.episodes % self.weight_sync_freq == 0:
            updates, aq_batches = ray.get(self.reward_learner.sync_weights.remote(self.aq_batches))
            if updates:
                self.regressor.model.load_state_dict(updates['weights'])
                self.reward_learner_logs = updates['logs']
                self.train_len = updates['train_len']
                # self.regressor.train_loader = ray.get(self.reward_learner.get_dataset.remote())
                self.aq_batches = aq_batches
                print('synced weights')

    def _get_dockscore(self, molecule):
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        loader = DataLoader([graph])
        mean, var = self.regressor.get_mean_variance(loader, self.train_len)
        # import pdb; pdb.set_trace();
        return (mean + (self.kappa * var)).item(), {'proxy_reward_mean': mean, 'proxy_reward_var': var}

    def _discount(self, molecule):
        mol = molecule.mol
        # QED constraint
        qed = QED.qed(mol)
        qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
        qed_discount = min(max(0.0, qed_discount), 1.0) # relu to maxout at 1

        # Synthesizability constraint
        synth = self.synth_net(mol=mol)
        synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
        synth_discount = min(max(0.0, synth_discount), 1.0) # relu to maxout at 1

        # Binding energy prediction
        self.reward_learner.add_molecule.remote(molecule, synth_discount * qed_discount)
        # dockscore, log_vals = self._get_dockscore(molecule)
        dockscore_reward, log_vals = self._get_dockscore(molecule)
        
        dockscore = self._simulation(mol=mol)
        # dockscore = (self.dockscore_std[0] - dockscore) / (self.dockscore_std[1])  # normalize against std dev
        if self.reward_learner_logs is not None:
            log_vals = { **log_vals, **self.reward_learner_logs}
            self.reward_learner_logs = None
        # combine rewards
        disc_reward = dockscore_reward * qed_discount * synth_discount
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, {"dockscore_reward": dockscore_reward, "chemprop_reward": dockscore, "qed": qed, "synth": synth, **log_vals}

    def _simulation(self, mol):
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        batch = Batch.from_data_list([graph]).to(self.device)
        pred = self.net(batch)
        reward = -float(pred.detach().cpu().numpy())
        return reward

    def __call__(self, molecule, simulate, env_stop, num_steps):
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
        self.dock = LambdaZero.chem.Dock_smi(tmp_dir,
                                  osp.join(programs_dir, 'chimera'),
                                  osp.join(programs_dir, 'dock6'),
                                  osp.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup"),
                                  gas_charge=True)
        self.target_norm = [-43.042, 7.057]
        self.attribute = attribute

    def eval(self, mol):
        s = Chem.MolToSmiles(mol[1])
        print("starting", s)
        try:
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        setattr(mol[0], 'gridscore', th.FloatTensor([-(r - self.target_norm[0]) / self.target_norm[1]]))
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        print("done", s, r)
        return mol, reward


@ray.remote(num_gpus=1, num_cpus=4)
class BayesianRewardActor():
    def __init__(self, config, use_dock_sim, binding_model, 
                    pretrained_model=None, transform=T.Compose([LambdaZero.utils.Complete()])):
        
        self.device = config["device"]
        self.transform = transform
        self.config = config
        self.regressor_config = config["regressor_config"]
        self.unseen_molecules = []
        self.train_molecules = []
        self.mols = 0
        self.use_dock_sim = use_dock_sim
        self.num_threads = 8
        self.max_size = 25000
        # self.sumtree = SumTree(self.max_size)
        self.train_len = 0
        self.prune_factor = 0.2

        self.regressor = config["regressor"](self.regressor_config)
        if use_dock_sim:
            self.actors = [_SimDockLet.remote(os.environ['SLURM_TMPDIR'], programs_dir, datasets_dir, self.config["data"]["target"])
                       for i in range(self.num_threads)]
            self.pool = ray.util.ActorPool(self.actors)
            
        else:
            self.net = LambdaZero.models.MPNNet()
            self.net.to(config['device'])
            self.net.load_state_dict(th.load(binding_model, map_location=th.device(config['device'])))
            self.net.eval()
        self.target_norm = [-43.042, 7.057]

        print('BR: Loaded Oracle Network')
        print('BR: Loading Dataset ...')

        self.dataset = config["data"]["dataset"](**config["data"]["dataset_config"])
        ul_idxs, val_idxs, test_idxs = np.load(self.config["data"]["dataset_split_path"], allow_pickle=True)

        np.random.shuffle(ul_idxs) # randomly acquire batch zero
        train_idxs = ul_idxs[:self.config["aq_size0"]]
        ul_idxs = ul_idxs[self.config["aq_size0"]:]
        train_set = Subset(self.dataset, train_idxs.tolist())
        self.ul_set = Subset(self.dataset, ul_idxs.tolist())
        val_set = Subset(self.dataset, val_idxs.tolist())
        self.train_len += self.config["aq_size0"]
        train_molecules = []
        for i, data in enumerate(train_set):
            setattr(data, config["data"]["target"], -(getattr(data, config["data"]["target"]) - self.target_norm[0]) / self.target_norm[1])
            train_molecules.append(data)
        
        self.train_loader = DataLoader(train_molecules, shuffle=True, batch_size=config["data"]["b_size"])
        self.val_loader = DataLoader(val_set, batch_size=config["data"]["b_size"])
        self.ul_loader = DataLoader(self.ul_set, batch_size=config["data"]["b_size"])
        self.dataset = None
        self.train_molecules = train_molecules
        print('BR: Loaded Dataset')
        
        if pretrained_model is not None:
            self.regressor.model.load_state_dict(th.load(pretrained_model))
            self.regressor.model = self.regressor.model.to(device)
            print('Loaded pretrained model')
        else:
            print('BR: Running single pretraining step ...')
            scores, val = self.regressor.fit(self.train_loader, self.val_loader)
        print(scores, val)
        self.logs = {**scores, **val}
        print('BR: Loaded Bayesian Reward Actor')
        self.batches = 0
    
    def prune(self):
        new_mols_idx = np.argsort([i[2] for i in self.unseen_molecules])[int(self.prune_factor * len(self.unseen_molecules)):]
        # new_sum_tree = SumTree(self.max_size)
        new_mols = []
        for i, j in enumerate(new_mols_idx):
            new_mols.append(self.mols[j])
        self.unseen_molecules = new_mols

    def _simulation(self, mol):
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        batch = Batch.from_data_list([graph]).to(self.device)
        pred = self.net(batch)
        reward = -float(pred.detach().cpu().numpy())
        return reward

    def get_weights(self):
        return self.regressor.model.state_dict()

    def get_logs(self):
        return self.logs

    def get_dataset(self):
        return self.train_loader

    def sync_weights(self, aq_batches):
        if self.batches > aq_batches:
            return {
                'weights': self.regressor.model.state_dict(),
                'logs': self.logs,
                'train_len': self.train_len
                }, self.batches
        return None, None

    def add_molecule(self, molecule, discount):
        if self.mols > self.max_size:
            self.prune()
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        self.unseen_molecules.append([graph, mol, None, discount])
        self.mols += 1
        if self.mols % self.config['num_mol_retrain'] == 0:
            print('Collected {} mols, performing batch acquisition'.format(self.mols))
            self.train_unseen_mols = self.unseen_molecules
            self.unseen_molecules = []
            self.train()

    def acquire_batch(self):
        mols = np.array(self.train_unseen_mols)[:, 0]
        loader = DataLoader(mols, batch_size=self.config["data"]["b_size"], num_workers=2, pin_memory=True)
        mean, var = self.regressor.get_mean_variance(loader, len(self.train_loader.dataset))
        scores = mean + (self.config["kappa"] * var)
        for i in range(len(self.train_unseen_mols)):
            # print(scores[i], self.train_unseen_mols[i][3])
            scores[i] = self.train_unseen_mols[i][3] * scores[i]
            self.train_unseen_mols[i][2] = self.train_unseen_mols[i][3] * scores[i]

        # noise is not a part of original UCT but is added here
        if self.config["minimize_objective"]: scores = -scores
        # import pdb; pdb.set_trace();
        idxs = np.argsort(-scores)[:self.config["aq_size"]]
        return idxs

    def compute_targets(self, mol_set):
        if self.use_dock_sim:
            n = len(mol_set)
            t0 = time.time()
            result = np.array(list(self.pool.map(lambda a, m: a.eval.remote(m), mol_set)))
            mol_set = np.array([mol for mol in result[:,0]]).tolist()
            rs = np.array([mol for mol in result[:,1]])
            t1 = time.time()
            # import pdb; pdb.set_trace();
            print('Docking sim done in {}'.format(t1-t0))
            print('Mean R in batch: ', np.mean(rs))
        else:
            n = len(mol_set)
            rs = [None] * n
            for i, (graph, mol, _, _) in enumerate(mol_set):
                r = self._simulation(mol)
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
        
        aq_mask = np.zeros(len(self.ul_set), dtype=np.bool)
        idx = np.random.choice(len(self.ul_set), len(idxs), replace=False)
        aq_mask[idx] = True
        aq_idxs = np.asarray(self.ul_set.indices)[aq_mask].tolist()
        ul_idxs = np.asarray(self.ul_set.indices)[~aq_mask].tolist()
        aq_set = Subset(self.ul_set, aq_idxs)
        self.ul_set = Subset(self.ul_set, ul_idxs)

        loader = DataLoader(np.array(aq_idx)[:,0], shuffle=False, batch_size=self.config["data"]["b_size"])
        mean, var = self.regressor.get_mean_variance(loader, len(self.train_loader.dataset))
        mse_before = np.mean((rews - mean) ** 2)

        fine_tune_dataset = aq_idx + [[graph, None, None, None] for graph in aq_set]
        fine_tune_loader = DataLoader(np.array(fine_tune_dataset)[:,0], shuffle=True, batch_size=self.config["data"]["b_size"])

        self.train_molecules.extend(np.array(fine_tune_dataset)[:, 0].tolist())
        
        self.train_loader = DataLoader(self.train_molecules, shuffle=True, batch_size=self.config["data"]["b_size"])
        self.unseen_molecules.extend(self.train_unseen_mols)

        # fit model to the data
        if self.batches % 5 == 0:
            scores, val = self.regressor.fine_tune(fine_tune_loader, self.train_loader, self.val_loader, True)
        else:
            scores, val = self.regressor.fine_tune(fine_tune_loader, self.train_loader, self.val_loader, False)
        
        self.train_len += self.config["aq_size"]
        mean, var = self.regressor.get_mean_variance(loader, len(self.train_loader.dataset))
        mse_after = np.mean((rews - mean) ** 2)

        for i in range(len(aq_idx)):
            rews[i] = rews[i] * aq_idx[i][3]

        self.logs = {**scores, **val, 'mse_before': mse_before, 'mse_after': mse_after, 'mse_diff': mse_before - mse_after, 
                        'oracle_max': np.max(rews), 'oracle_mean': np.mean(rews)}
        print(self.logs)

        logging.info(scores)
        return scores

    def train(self):
        print('BR: Acquring Batch ...')
        idx = self.acquire_batch()
        print('BR: Acquired Batch')
        print('BR: Retraining with Acquired Batch ...')
        self.update_with_seen(idx)
        print('Bayesian Reward Actor: Retrained')
        self.batches += 1


class QEDReward:
    def __init__(self):
        pass
    def __call__(self, molecule, done, num_steps):
        mol = molecule.mol
        if mol is None:
            return 0.0, {"discounted_reward": 0.0, "QED": 0.0}
        qed = QED.qed(mol)
        if done:
            discounted_reward = qed
        else:
            discounted_reward = 0.0

        return discounted_reward, {"discounted_reward": discounted_reward, "QED": qed}

class MorganDistReward:
    def __init__(self, target, fp_len, fp_radius, limit_atoms):
        self.fp_len, self.fp_radius = fp_len, fp_radius
        self.limit_atoms = limit_atoms
        target = Chem.MolFromSmiles(target)
        self.target_fp = AllChem.GetMorganFingerprintAsBitVect(target, self.fp_radius, self.fp_len)
    def __call__(self, molecule, done, num_steps):
        mol = molecule.mol
        if mol is not None:
            natm = mol.GetNumAtoms()
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.fp_radius, self.fp_len)
            reward = DataStructs.DiceSimilarity(self.target_fp, fp)
            if natm < self.limit_atoms[0]:
                discounted_reward = reward
            else:
                natm_discount = max(0.0, self.limit_atoms[1] - natm) / (self.limit_atoms[1] - self.limit_atoms[0])
                discounted_reward = reward * natm_discount
        else:
            reward, discounted_reward = 0.0, 0.0
        return reward, discounted_reward
