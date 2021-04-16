import torch as th
from torch_geometric.data import Batch, DataLoader

from rdkit import Chem
from rdkit.Chem import QED, AllChem
from rdkit import DataStructs
from torch.utils.data import Subset, ConcatDataset
import torch_geometric.transforms as T
import LambdaZero.utils
import LambdaZero.models
import LambdaZero.chem
from LambdaZero.utils import Complete, get_external_dirs
from LambdaZero.examples.bayesian_models.bayes_tune.deep_ensemble import DeepEnsemble
from LambdaZero.examples.bayesian_models.bayes_tune.mcdrop import MCDrop

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
                discounted_reward, log_vals = -0.5, {"dock_reward":-0.5, "natm": 0.0, "qed": -0.5, "synth": -0.5}
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


class QEDReward:
    def __init__(self, **kwargs):
        pass

    def reset(self, previous_reward=0.0):
        pass

    def __call__(self, molecule, simulate , done, num_steps):
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


class SynthQEDReward:
    def __init__(self, qed_cutoff, synth_cutoff, synth_config, **kwargs):
        self.qed_cutoff = qed_cutoff
        self.synth_cutoff = synth_cutoff
        self.synth_net = LambdaZero.models.ChempropWrapper_v1(synth_config)

    def reset(self, previous_reward=0.0):
        pass

    def __call__(self, molecule, simulate , done, num_steps):
        mol = molecule.mol
        if (mol is not None) and (len(molecule.jbonds) > 0):
            qed = QED.qed(mol)
            qed_discount = (qed - self.qed_cutoff[0]) / (self.qed_cutoff[1] - self.qed_cutoff[0])
            qed_discount = min(max(0.0, qed_discount), 1.0)  # relu to maxout at 1

            # Synthesizability constraint
            synth = self.synth_net(mol=mol)
            synth_discount = (synth - self.synth_cutoff[0]) / (self.synth_cutoff[1] - self.synth_cutoff[0])
            synth_discount = min(max(0.0, synth_discount), 1.0)  # relu to maxout at 1
            discounted_reward = qed_discount * synth_discount

            return discounted_reward, {"discounted_reward": discounted_reward, "QED": qed, "synthesizability": synth}
        else: return 0.0, {"discounted_reward": 0.0, "QED": 0.0}
