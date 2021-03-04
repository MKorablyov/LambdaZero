import pandas as pd
import os.path as osp
import random

from rdkit import Chem
from rdkit.Chem import MolStandardize, QED

import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger, WandbTrainableMixin
from LambdaZero.contrib.htp_chemistry import mc_sampling_config as config

import wandb
# from LambdaZero.contrib.loggers import WandbRemoteLoggerCalback, RemoteLogger
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

'''
we consider common, efficient, (mostly) non-interferring reactions that focus on cross-couplings (extension of molecules)
the goal is as few as possible, and reuses the same functional groups as frequently as possible

functional groups include:
1. carboxylic acid
2. carbonyl
3. alkyl alcohol
4. alkyl amine
6. alkyl halide
7. aryl halide
8. aryl boronic acid (pinacol ester)
 
reactions considered include:
1. amide formation (carboxylic acid + amine)
2. reductive amination (carbonyl + amine)
3. esterification (carboxylic acid + alkyl halide or alcohol)
4. Suzuki coupling (aryl halide + aryl boronic acid (ester)) 
5. Buchwald–Hartwig amination (aryl halide + amine)

other considered ones could include: 
1. SnAr 
2. phenol alkylation
3. Aldol condensation
4. alcohol oxidation
5. aromatic halogenation (if reagents does not exist)

we should also consider protection groups
'''

class MC_sampling_v0():

    def __init__(self, config):
        self.mols = list(pd.read_csv(config["mols"], sep=r'\\t', header=None, engine='python')[0])

        self.bicomponent_reactions = config["bicomponent_reactions"]()
        self.monocomponent_reactions = config["monocomponent_reactions"]()

        self.num_reaction_steps = config["num_reaction_steps"]
        self.qed_cutoff = config["qed_cutoff"]
        self.multi_substitution = config["multi_substitution"]


        self.normalizer = MolStandardize.normalize.Normalizer()
        self.lfc = MolStandardize.fragment.LargestFragmentChooser()
        self.uc = MolStandardize.charge.Uncharger()

    def salt_remover(self, mol):
        mol = self.normalizer.normalize(mol)
        mol = self.lfc.choose(mol)
        mol = self.uc.uncharge(mol)
        return mol

    def product(self, mol1, mol2):
        products = []
        for reaction in self.bicomponent_reactions: #["two_components"]:
            if self.multi_substitution:
                rxn_product = ((0,),)
                step_product = None
                step = 0
                while len(rxn_product) > 0:
                    if step == 0:
                        rxn_product = reaction.RunReactants((mol1, mol2), maxProducts=50) \
                                        + reaction.RunReactants((mol2, mol1), maxProducts=50)
                    else:
                        rxn_product = reaction.RunReactants((step_product, mol2), maxProducts=50) \
                                       + reaction.RunReactants((step_product, mol1), maxProducts=50) \
                                       + reaction.RunReactants((mol2, step_product), maxProducts=50) \
                                       + reaction.RunReactants((mol1, step_product), maxProducts=50)
                    if len(rxn_product) > 0:
                        step_product = rxn_product[-1][0]
                        # print(Chem.MolToSmiles(step_product), Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2))
                        step_product.UpdatePropertyCache()
                    step += 1
                    if step >= 5:
                        # possibly a polymer material
                        break
                if step_product is not None:
                    products.append(step_product)
            else:
                # if we only care about monosubstution:
                rxn_product = reaction.RunReactants((mol1, mol2), maxProducts=50) + \
                               reaction.RunReactants((mol2, mol1), maxProducts=50)
                if len(rxn_product) > 0:
                    step_product = rxn_product[-1][0]
                    step_product.UpdatePropertyCache()
                    products.append(step_product)

        for idx, reaction in enumerate(self.monocomponent_reactions):
            if self.multi_substitution:
                rxn_product = ((0,),)
                step_product = None
                step = 0
                while len(rxn_product) > 0:
                    if step == 0:
                        rxn_product = reaction.RunReactants((mol1, ), maxProducts=50)
                    else:
                        rxn_product = reaction.RunReactants((mol1, ), maxProducts=50)
                    if len(rxn_product) > 0:
                        step_product = rxn_product[-1][0]
                        step_product.UpdatePropertyCache()
                    step += 1
                    if step >= 5:
                        break
                if step_product is not None:
                    products.append(step_product)
            else:
                rxn_product = reaction.RunReactants((mol1, ), maxProducts=50)
                if len(rxn_product) > 0:
                    step_product = rxn_product[-1][0]
                    try:
                        step_product.UpdatePropertyCache()
                    except Exception as e:
                        print(idx, e, Chem.MolToSmiles(mol1), 'opps\n')
                    products.append(step_product)

        if len(products) > 0:
            return random.choice(products)
        else:
            return None

    def sample_mol(self):
        mols = random.choices(self.mols, k=(self.num_reaction_steps + 1))  # steps = 1, sample 2 mols
        # print(mols)
        # mols = ['COCCn1cc(nn1)C(=O)O', 'NS(=O)(=O)NC1CCOCC1']
        mols = [self.salt_remover(Chem.MolFromSmiles(mol)) for mol in mols]
        prev_mol = mols[0]
        product = None
        for i in range(1, len(mols)):
            current_mol = mols[i]
            product = self.product(prev_mol, current_mol)
            if product is None:
                return None
            prev_mol = product

        smiles = Chem.MolToSmiles(product)
        natm = product.GetNumAtoms()

        if self.qed_cutoff is not None: # additional filter
            if QED.qed(product) >= self.qed_cutoff:
                return [product, smiles, natm, QED.qed(product)]
            else: return None
        else:
            return [product, smiles, natm, QED.qed(product)]


# tune.Trainable version of v0, only for generating mols/convergence
class MC_sampling_v1(WandbTrainableMixin, tune.Trainable):

    def setup(self, config):
        self.mc_sampling = MC_sampling_v0(config)
        self.convergence_criteria = config["convergence_criteria"]
        self.convergence_check_frequency = config["convergence_check_frequency"]
        self.update_frequency = config["update_frequency"]
        self.generate_molecules_mode = config["generate_molecules_mode"]

        self.valid_mols = []
        self.total_mol = 0.
        self.current_ratio = 0.
        self.previous_ratio = 1.
        self.convergence = 1.

    def step(self):
        for i in range(self.update_frequency):
            product = self.mc_sampling.sample_mol()
            self.total_mol += 1
            if product is not None:
                self.valid_mols.append(product[1:4])
            self.current_ratio = len(self.valid_mols) / self.total_mol
            self.convergence = abs(self.current_ratio - self.previous_ratio)
            self.previous_ratio = self.current_ratio
            wandb.log({"convergence": self.convergence, "valid_mol_ratio": self.current_ratio, "num_valid_mols": len(self.valid_mols)})

        if self.generate_molecules_mode is None: # stop at convergence criteria
            if (self._iteration+1) % self.convergence_check_frequency == 0:
                if self.convergence <= self.convergence_criteria:
                    self.handcraft_save_checkpoint()
        elif len(self.valid_mols) >= self.generate_molecules_mode: # generate n molecules
            self.handcraft_save_checkpoint()

        return {"convergence": self.convergence, "valid_mol_ratio": self.current_ratio, "num_valid_mols": len(self.valid_mols)}

    # def save_checkpoint(self, tmp_checkpoint_dir):
    #     checkpoint_path = osp.join(tmp_checkpoint_dir, str(self.num_reaction_steps)+"_valid_mols.csv")
    #     products = pd.DataFrame(self.valid_mols)
    #     products.to_csv(checkpoint_path, index=False)
    #     # wandb.log({"molecules": wandb.Table(dataframe=products)})
    #     return tmp_checkpoint_dir

    def handcraft_save_checkpoint(self):
        checkpoint_path = osp.join(self.logdir, str(self.num_reaction_steps) + "_valid_mols.csv")
        products = pd.DataFrame.from_records(self.valid_mols, columns=["smiles", "natm", "qed"])
        products = products.drop_duplicates()
        products.to_csv(checkpoint_path, index=False)
        wandb_products = wandb.Table(dataframe=products)
        wandb.log({"molecules": wandb_products})
        import matplotlib.pyplot as plt
        plt.scatter(products.natm, products.qed, alpha=0.5)
        plt.xlabel("Number of atoms")
        plt.ylabel("QED")
        wandb.log({"molecule_properties": plt})
        plt.clf()
        # wandb.log({"molecule_properties": wandb.plot.scatter(wandb_products, "qed", "natm", title='Molecular properties')})
        return None

def mc_sampling_stopper(trial_id, result):
    if result['config']['generate_molecules_mode'] is None:
        if result['training_iteration'] % result['config']['convergence_check_frequency'] == 0:
            if result['convergence'] <= result['config']['convergence_criteria']:
                return True
    # if used to generate molecules:
    elif result['num_valid_mols'] >= result['config']['generate_molecules_mode']:
        return True
    return False

# if len(sys.argv) >= 2: config_name = sys.argv[1]
# else:
config_name = "mc_sampling_config_002"
config = getattr(config,config_name)

DEFAULT_CONFIG = {
    "mc_sampling_config": config,
    "summaries_dir": summaries_dir,
    "memory": 5*10**9, # 30 for computecanada
    "object_store_memory": 5*10**9,
    "resources_per_trial": {"cpu": 12},
    "checkpoint_at_end": False, # True if using the default save_checkpoint, else it is a handcrafted function
    "stop": mc_sampling_stopper,
    "num_samples": 1,
    "logger":DEFAULT_LOGGERS + (WandbLogger, ),
}

if __name__ == "__main__":

    config = DEFAULT_CONFIG
    ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])

    # # initialize loggers, note to log mols and graphs the handcrafted savingpoint is more efficient
    # os.environ['WANDB_DIR'] = summaries_dir
    # remote_logger = RemoteLogger.remote()
    # wandb_callback = WandbRemoteLoggerCalback(
    #     remote_logger=remote_logger,
    #     project="htp_chemistry",
    #     api_key_file=osp.join(summaries_dir,"wandb_key"),
    #     log_config=False)

    tune.run(MC_sampling_v1,
        config=config["mc_sampling_config"],
        stop=config["stop"],
        local_dir=summaries_dir,
        name="htp_chemistry",
        checkpoint_at_end=config["checkpoint_at_end"],
        resources_per_trial=config["resources_per_trial"],
        num_samples=config["num_samples"],
        # loggers = DEFAULT_LOGGERS + (wandb_callback,),
    )
