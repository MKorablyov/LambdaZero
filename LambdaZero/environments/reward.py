import torch as th
from torch_geometric.data import Batch, DataLoader
import numpy as np
import copy
import ray
import time
import os.path as osp
import os
import threading
from sklearn import linear_model
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
        discount = natm_discount * qed_discount
        disc_reward = min(reward, reward * discount) # don't appy to negative rewards
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, qed, discount

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

        discount = 0
        if simulate:
            reward = self._simulation(molecule)
            if reward is not None:
                discounted_reward, qed, discount = self._discount(molecule.mol, reward)
            else:
                reward, discounted_reward, qed = -0.5, -0.5, -0.5
        else:
            reward, discounted_reward, qed = 0.0, 0.0, 0.0
        return discounted_reward, {"reward": reward, "discounted_reward": discounted_reward, "QED": qed, "discount": discount}

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
                 soft_stop, exp, delta, simulation_cost, device, transform=T.Compose([LambdaZero.utils.Complete()]), **unused):

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
        dockscore_normalized = (self.dockscore_std[0] - dockscore) / (self.dockscore_std[1])  # normalize against std dev

        # combine rewards
        discount = qed_discount * synth_discount
        disc_reward = dockscore_normalized * discount
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, {"dockscore": dockscore_normalized, "qed": qed, "synth": synth,
                             "discount": discount}

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
        ul_idxs = np.asarray(self.train_unseen_mols)[~aq_mask].tolist()
        self.train_unseen_mols = np.asarray(self.train_unseen_mols)[~aq_mask].tolist()

        aq_idx, rews = self.compute_targets(aq_idxs)
        ul_idx, _ = self.compute_targets(ul_idxs)
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

        ul_tune_dataset = ul_idx + [[graph, None, None, None] for graph in self.ul_set]
        ul_tune_loader = DataLoader(np.array(ul_tune_dataset)[:,0], shuffle=True, batch_size=self.config["data"]["b_size"])

        self.train_molecules.extend(np.array(fine_tune_dataset)[:, 0].tolist())
        
        self.train_loader = DataLoader(self.train_molecules, shuffle=True, batch_size=self.config["data"]["b_size"])
        self.unseen_molecules.extend(self.train_unseen_mols)

        expected_idxs = self.thompson_acquire_batch(seen = fine_tune_loader, unseen = ul_tune_loader, size = self.config['aq_size'])
        th_aq_mask = np.zeros(len(self.ul_set.dataset), dtype=np.bool)
        th_aq_mask[expected_idxs] = True
        th_aq_idxs = th_aq_mask.nonzero()[0].tolist()
        th_ul_idxs = (th_aq_mask == False).nonzero()[0].tolist()
        th_aq_set = Subset(self.ul_set, th_aq_idxs)
        th_ul_set = Subset(self.ul_set, th_ul_idxs)
        th_train_set  = DataLoader(th_aq_set, shuffle=False, batch_size=1)
        th_ul_loader = DataLoader(th_ul_set, shuffle=False, batch_size=1)

        """
        Current problem: the th_aq_set and the th_ul_set subsets seem to be empty.
        th_aq_idxs is a list of 32 indices. If one of the elements in the list is 256,
        256 will be present in self.ul_set.indices. However, trying to print self.ul_set[256] will
        yield an IndexError. This occurs for all of the indices in th_aq_idxs. The same thing happens
        with th_ul_idxs and the th_ul_loader. They loaders have a len but return nothing when trying to
        iterate over them. Even next(iter(th_ul_loader)) returns index out of range.
        """

        expected_scores = self.th_aq_regret(th_train_set, th_ul_loader, self.config['data']['target'])

        # fit model to the data
        if self.batches % 5 == 0:
            scores, val = self.regressor.fine_tune(fine_tune_loader, self.train_loader, self.val_loader, True)
        else:
            scores, val = self.regressor.fine_tune(fine_tune_loader, self.train_loader, self.val_loader, False)
        
        self.ul_set = Subset(self.ul_set, ul_idxs)

        self.train_len += self.config["aq_size"]
        mean, var = self.regressor.get_mean_variance(loader, len(self.train_loader.dataset))
        mse_after = np.mean((rews - mean) ** 2)

        for i in range(len(aq_idx)):
            rews[i] = rews[i] * aq_idx[i][3]

        self.logs = {**scores, **val, **expected_scores, 'mse_before': mse_before, 'mse_after': mse_after, 'mse_diff': mse_before - mse_after, 
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

    def thompson_acquire_batch(self, seen, unseen, size=500):
        idxs = []
        try:
            for _ in range(size):
                embs = []
                y = np.concatenate([getattr(d, 'gridscore').cpu().numpy() for d in seen.dataset],0)
                for bidx, data in enumerate(seen):
                    data = data.to(device)
                    emb = self.regressor.model.get_embed(data, False)
                    embs.append(emb.detach().cpu().numpy())
                embs = np.concatenate(embs, 0)

                ul_embs = []

                for bidx, data in enumerate(unseen):
                    data = data.to(device)
                    emb = self.regressor.model.get_embed(data, False)
                    ul_embs.append(emb.detach().cpu().numpy())
                ul_embs = np.concatenate(ul_embs, 0)

                brr = linear_model.BayesianRidge()
                brr.fit(embs, y)
                out = brr.predict(ul_embs)
                y_star = out.argsort()
                for i in y_star:
                    if i not in idxs:
                        idx = i
                        break
                
                # score = self.update_with_seen(idx)
                # idx = self.get_posterior_sample(X_in = ul_embs,X_trn = embs, y = y, idxs = idxs, X_feature_size = 64, sigma_y = 0.2)
                if idx: idxs.append(int(idx))
        except:
            pdb.set_trace()
        return idxs

    # def get_posterior_sample(self, X_in, X_trn, y, idxs, X_feature_size,sigma_y):

    #     w_0 = np.zeros(X_feature_size)
    #     V_0 = np.diag([sigma_y] * X_feature_size)**2
    #     V0_inv = np.linalg.inv(V_0)
    #     V_n = sigma_y**2 * np.linalg.inv(sigma_y**2 * V0_inv + (X_trn.T @ X_trn))
    #     w_n = V_n @ V0_inv @ w_0 + 1 / (sigma_y**2) * V_n @ X_trn.T @ y

    #     #sample w from distribution N(w_n,V_n)
    #     samples = np.random.multivariate_normal(w_n, V_n, size = 1)
    #     #pick the argmax of expectation of y* which is x*w
    #     y_star = X_in @ samples.T
    #     #add best the train set
    #     y_star = y_star.argsort()
    #     for i in y_star:
    #         if i not in idxs:
    #             return i

    def th_aq_regret(self, train_loader, ul_loader, target_name):
        # import pdb; pdb.set_trace()
        # try:
        train_targets = np.concatenate([d['gridscore'].cpu().numpy() for d in train_loader.dataset], 0)#np.concatenate([getattr(d, config["data"]["target"]).cpu().numpy() for d in train_loader.dataset],0)
        ul_targets = np.concatenate([d['gridscore'].cpu().numpy() for d in ul_loader.dataset], 0)#np.concatenate([getattr(d, config["data"]["target"]).cpu().numpy() for d in ul_loader.dataset],0)
        all_targets = np.concatenate([train_targets, ul_targets],0)
        train_sorted = train_targets[np.argsort(train_targets)]
        all_sorted = all_targets[np.argsort(all_targets)]

        top15_regret = np.median(train_sorted[:15]) - np.median(all_sorted[:15])
        top50_regret = np.median(train_sorted[:50]) - np.median(all_sorted[:50])
        aq_top15 = np.median(train_sorted[:15])
        aq_top50 = np.median(train_sorted[:15])

        n = int(all_targets.shape[0] * 0.01)
        frac_top1percent = np.asarray(train_sorted[:n] <= all_sorted[n],dtype=np.float).mean()
        return {"th_aq_top15_regret":top15_regret, "th_aq_top50_regret":top50_regret, "th_aq_top15":aq_top15, 
                "th_aq_top50":aq_top50, "th_aq_frac_top1_percent":frac_top1percent}
        # except:
        #     import pdb; pdb.set_trace()


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
