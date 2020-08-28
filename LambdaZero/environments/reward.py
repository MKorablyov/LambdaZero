import torch as th
from torch_geometric.data import Batch, DataLoader
import numpy as np
import copy
import ray
from rdkit import Chem
from rdkit.Chem import QED, AllChem
from rdkit import DataStructs
from torch.utils.data import Subset, ConcatDataset
import torch_geometric.transforms as T
import LambdaZero.utils
import LambdaZero.models
import LambdaZero.chem
from LambdaZero.utils import Complete, get_external_dirs
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

def aq_regret(train_loader, ul_loader, config):
    train_targets = np.concatenate([getattr(d, config["data"]["target"]).cpu().numpy() for d in train_loader.dataset],0)
    ul_targets = np.concatenate([getattr(d, config["data"]["target"]).cpu().numpy() for d in ul_loader.dataset],0)
    all_targets = np.concatenate([train_targets, ul_targets],0)
    train_sorted = train_targets[np.argsort(train_targets)]
    all_sorted = all_targets[np.argsort(all_targets)]

    top15_regret = np.median(train_sorted[:15]) - np.median(all_sorted[:15])
    top50_regret = np.median(train_sorted[:50]) - np.median(all_sorted[:50])
    aq_top15 = np.median(train_sorted[:15])
    aq_top50 = np.median(train_sorted[:15])

    n = int(all_targets.shape[0] * 0.01)
    frac_top1percent = np.asarray(train_sorted[:n] <= all_sorted[n],dtype=np.float).mean()
    return {"aq_top15_regret":top15_regret, "aq_top50_regret":top50_regret, "aq_top15":aq_top15, "aq_top50":aq_top50,
            "aq_frac_top1_percent":frac_top1percent}

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
                pca = LambdaZero.utils.molecule_pca(molecule.mol)
                log_vals = {**pca, **log_vals}
            else:
                discounted_reward, log_vals = 0.0, {}
        else:
            discounted_reward, log_vals = 0.0, {}

        return discounted_reward, log_vals

class PredDockBayesianReward_v1:
    def __init__(self, qed_cutoff, synth_config, dockscore_config,
                 soft_stop, exp, delta, simulation_cost, reward_learner, device, 
                 transform=T.Compose([LambdaZero.utils.Complete()])):

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
        # self.binding_net = LambdaZero.models.ChempropWrapper_v1(dockscore_config)
        self.reward_learner = reward_learner

    def reset(self, previous_reward=0.0):
        self.previous_reward = previous_reward

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
        self.reward_learner.add_molecule.remote(molecule)
        dockscore, log_vals = ray.get(self.reward_learner.get_reward.remote(molecule))
        # dockscore = self.binding_net(mol=mol)
        dockscore_reward = (self.dockscore_std[0] - dockscore) / (self.dockscore_std[1])  # normalize against std dev

        # combine rewards
        disc_reward = dockscore_reward * qed_discount * synth_discount
        if self.exp is not None: disc_reward = self.exp ** disc_reward

        # delta reward
        delta_reward = (disc_reward - self.previous_reward - self.simulation_cost)
        self.previous_reward = disc_reward
        if self.delta: disc_reward = delta_reward
        return disc_reward, {"dockscore_reward": dockscore_reward, "qed": qed, "synth": synth, **log_vals}

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

@ray.remote(num_gpus=1, num_cpus=3)
class BayesianRewardActor():
    def __init__(self, config, regressor, regressor_config, dockscore_config, device, 
                    transform=T.Compose([LambdaZero.utils.Complete()])):
        self.device = device
        self.transform = transform
        self.config = config
        self.regressor_config = regressor_config
        self.unseen_molecules = []
        self.train_molecules = []
        self.mols = 0
        # self.init = False
        self.regressor = regressor(regressor_config)
        # self.dock = LambdaZero.chem.Dock_smi(tmp_dir,
        #                           osp.join(programs_dir, 'chimera'),
        #                           osp.join(programs_dir, 'dock6'),
        #                           osp.join(datasets_dir, 'brutal_dock/d4/docksetup/'))
        self.target_norm = [-26.3, 12.3]
        self.binding_net = LambdaZero.models.ChempropWrapper_v1(dockscore_config)
        print('BR: Loaded Oracle Network')
        print('BR: Loading Dataset ...')

        self.dataset = config["data"]["dataset"](**config["data"]["dataset_config"])
        ul_idxs, val_idxs, test_idxs = np.load(self.config["data"]["dataset_split_path"], allow_pickle=True)

        np.random.shuffle(ul_idxs) # randomly acquire batch zero
        train_idxs = ul_idxs[:self.config["aq_size0"]]
        train_set = Subset(self.dataset, train_idxs.tolist())
        val_set = Subset(self.dataset, val_idxs.tolist())
        # for i, data in enumerate(train_set):
        #     data.y = getattr(data, config["data"]["target"])
        #     self.train_molecules.append([data, None])
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=config["data"]["b_size"])
        self.val_loader = DataLoader(val_set, batch_size=config["data"]["b_size"])

        print('BR: Loaded Dataset')
        print('BR: Running single pretraining step ...')
        self.regressor.fit(self.train_loader,self.val_loader)
        # self.init == True
        print('BR: Loaded Bayesian Reward Actor')
        self.regressor_copy = regressor(regressor_config)
        self.regressor_copy.train_loader = self.train_loader
        self.regressor_copy.model.load_state_dict(self.regressor.model.state_dict())
        # self.net = LambdaZero.models.MPNNetDrop(**model_config)
        # self.net.to(device)
        # self.net.load_state_dict(th.load(load_model, map_location=th.device(device)))
        # self.net.eval()
    
    def add_molecule(self, molecule):
        # while self.init == False:
        #     continue
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        self.unseen_molecules.append([graph, mol])
        self.mols += 1
        if self.mols % self.config['num_mol_retrain'] == 0:
            print('Collected {} mols, performing batch acquisition'.format(self.mols))
            self.train_unseen_mols = self.unseen_molecules
            self.unseen_molecules = []
            self.train()

    def acquire_batch(self):
        mols = np.array(self.train_unseen_mols)[:, 0]
        loader = DataLoader(mols, shuffle=True, batch_size=self.config["data"]["b_size"])
        mean, var = self.regressor.get_mean_variance(loader)
        scores = mean + (self.config["kappa"] * var)
        # noise is not a part of original UCT but is added here
        noise = self.config["epsilon"] * np.random.uniform(size=mean.shape[0])
        scores = scores + noise
        if self.config["minimize_objective"]: scores = -scores
        idxs = np.argsort(-scores)[:self.config["aq_size"]]
        return idxs

    def compute_targets(self, mol_set):
        # def f(i):
        #     try:
        #         _, r, _ = self.dock.dock(Chem.MolToSmiles(mol_set[i][1]))
        #     except Exception as e: # Sometimes the prediction fails
        #         print(e)
        #         r = 0
        #     mols[i].y = -(r - self.target_norm[0])/self.target_norm[1]
        #     rs[i]
        # t0 = time.time()
        # threads = []
        # thread_at = self.num_threads
        # for i, (graph, mol) in enumerate(mol_set):
        #     threads.append(threading.Thread(target=f, args=(i)))
        #     if i < self.num_threads:
        #         threads[-1].start()
        # while None in rewards:
        #     if sum([i.is_alive() for i in threads]) < self.num_threads and thread_at < n:
        #         threads[thread_at].start()
        #         thread_at += 1
        #     time.sleep(0.5)
        # t1 = time.time()
        for i, (graph, mol) in enumerate(mol_set):
            graph.y = self.binding_net(mol=mol)
        # print('Docking sim done in {}'.format(t1-t0))
        return mol_set

    def update_with_seen(self, idxs):
        aq_mask = np.zeros(len(self.train_unseen_mols), dtype=np.bool)
        aq_mask[idxs] = True
        aq_idxs = np.asarray(self.train_unseen_mols)[aq_mask].tolist()
        aq_idx = self.compute_targets(aq_idxs)
        self.train_unseen_mols = np.asarray(self.train_unseen_mols)[~aq_mask].tolist()
        # import pdb; pdb.set_trace();
        self.train_molecules.extend(aq_idxs)
        self.train_loader = DataLoader(np.array(self.train_molecules)[:,0], shuffle=True, batch_size=self.config["data"]["b_size"])
        self.unseen_molecules.extend(self.train_unseen_mols)
        # self.ul_loader = DataLoader(np.array(self.unseen_molecules)[:,0], batch_size=self.config["data"]["b_size"])
        # fit model to the data

        scores = self.regressor.fit(self.train_loader, self.val_loader)[-1]
        # compute acquisition metrics
        # print('Compute Aq Metrics')
        # _scores = aq_regret(self.train_loader, self.ul_loader, self.regressor_config)
        # scores = {**scores, **_scores}
        print(scores)
        return scores

    def get_reward(self, molecule):
        # while self.init == False:
        #     continue
        mol = molecule.mol
        atmfeat, _, bond, bondfeat = LambdaZero.chem.mpnn_feat(mol, ifcoord=False)
        graph = LambdaZero.chem.mol_to_graph_backend(atmfeat, None, bond, bondfeat)
        graph = self.transform(graph)
        loader = DataLoader([graph])
        mean, var = self.regressor_copy.get_mean_variance(loader)
        # import pdb; pdb.set_trace();
        return (mean + (self.config["kappa"] * var)).item(), {'energy_mean': mean, 'energy_var': var}

    def train(self):
        print('BR: Acquring Batch ...')
        idx = self.acquire_batch()
        print('BR: Acquired Batch')
        print('BR: Retraining with Acquired Batch ...')
        self.update_with_seen(idx)
        print('Bayesian Reward Actor: Retrained')
        self.regressor_copy.model.load_state_dict(self.regressor.model.state_dict())


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
