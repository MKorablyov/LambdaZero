import os
import random
import string
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

# from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from sklearn.metrics import roc_curve, auc  # , roc_auc_score

from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, _ = get_external_dirs()


class GenMolFile_v1:
    def __init__(self, outpath, num_conf):
        self.outpath = outpath
        self.num_conf = num_conf

    def __call__(self, smi, mol_name):
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol_h, numConfs=self.num_conf)
        [AllChem.MMFFOptimizeMolecule(mol_h, confId=i) for i in range(self.num_conf)]
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
        # choose minimum energy conformer
        mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(self.num_conf)])
        mol = Chem.RemoveHs(mol_h)
        # save file in .mol format
        mol_file = os.path.join(self.outpath, mol_name + ".mol")
        print(Chem.MolToMolBlock(mol, confId=int(mi)), file=open(mol_file, 'w+'))
        return mol_file


class DockVina_smi:
    def __init__(self, config):
        self.config = config
        self.gen_molfile = config["gen_molfile"](**config["gen_molfile_par"])
        self.rec_file = os.path.join(self.config["docksetup_dir"], "rec.pdb")
        self.lig_file = os.path.join(self.config["docksetup_dir"], "lig.pdb")

        os.makedirs(self.config["outpath"], exist_ok=True)
        os.makedirs(self.config["gen_molfile_par"]["outpath"], exist_ok=True)

    def dock(self, smi, mol_name):
        mol_file = self.gen_molfile(smi, mol_name)                                  # make input .mol file to docking
        sminaord_file = os.path.join(self.config["outpath"], mol_name + ".pdb")     # filepath to output docked molecule (multiple conformations in the same file)
        dock_cmd = f"{self.config['smina_bin']} " \
                   f"-r {self.rec_file} " \
                   f"-l {mol_file} " \
                   f"--autobox_ligand {self.lig_file} " \
                   f"-o {sminaord_file} " \
                   f"{self.config['dock_pars']}"

        docking = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
        docking.wait()

    def extract_best_dockscore(self, mol_name):
        dockscore = float('nan')
        sminaord_file = os.path.join(self.config["outpath"], mol_name + ".pdb")
        try:
            with open(os.path.join(sminaord_file)) as f:
                smina_out = f.readlines()

            if smina_out[1].startswith("REMARK minimizedAffinity"):
                dockscore = float(smina_out[1].split(" ")[-1])
            else:
                raise Exception("Can't parse docking energy!")
        except Exception as e:
            print(e)
        finally:
            return dockscore

    def get_dockscore(self, smi, mol_name=None):
        mol_name = mol_name or ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        self.dock(smi, mol_name)
        return self.extract_best_dockscore(mol_name)


if __name__ == "__main__":
    config = {
        "outpath": os.path.join(datasets_dir, "seh/4jnc/docked"),
        "smina_bin": os.path.join(programs_dir, "smina/smina.static"),
        "docksetup_dir": os.path.join(datasets_dir, "seh/4jnc"),
        "dock_pars": "",  # "--exhaustiveness 8 --cpu 1",
        "gen_molfile": GenMolFile_v1,
        "gen_molfile_par": {
            "outpath": os.path.join(datasets_dir, "seh/4jnc/docked"),
            "num_conf": 20,
        }}

    data = pd.read_csv(os.path.join(datasets_dir, "seh/seh_chembl.csv"), sep=";")
    binding = data[["Smiles", "Standard Value", "Ligand Efficiency BEI", "Ligand Efficiency SEI"]].copy()
    binding["Standard Value"] = pd.to_numeric(binding["Standard Value"], errors="coerce")
    binding["Ligand Efficiency BEI"] = pd.to_numeric(binding["Ligand Efficiency BEI"], errors="coerce")
    binding["Ligand Efficiency SEI"] = pd.to_numeric(binding["Ligand Efficiency SEI"], errors="coerce")
    binding = binding.dropna()
    ic50 = binding["Standard Value"].to_numpy()
    binders = ic50 < 1
    decoys = ic50 > 10000
    binding = pd.concat([binding[binders], binding[decoys]])
    binding.reset_index(inplace=True)
    smis = binding["Smiles"].to_numpy()
    print(len(smis))

    dock_smi = DockVina_smi(config)
    dockscore = pd.DataFrame({"dockscore": [dock_smi.get_dockscore(smi) for smi in smis]})
    binding = pd.concat([binding, dockscore], axis=1)
    binding.to_feather(os.path.join(config["outpath"], "seh.ftr"))
    binding = pd.read_feather(os.path.join(config["outpath"], "seh.ftr"))
    # binding = pd.read_feather("/home/maksym/Datasets/seh/4jnc/docked/seh_v1.ftr")
    binding = binding.dropna()

    fpr, tpr, _ = roc_curve(binding["Standard Value"] < 1, -binding["dockscore"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.show()
    # print(roc_auc)
    # print(roc_auc_score(binding["Standard Value"] < 1, -binding["dockscore"]))
    # plt.scatter(np.log(binding["Standard Value"]), binding["dockscore"])
    # plt.show()

# import prody as pr

#     # make the ligand with the order of atoms shuffled how Vina wants it, but with heavy atoms being in the same place
#     reorder_cmd = smina_cmd + " --score_only" + " -o " + os.path.join(init.db_root, init.out_dir, uid + "_sminaord.pdb")
#     # make the actual docking command
#     dock_cmd = smina_cmd+init.dock_pars + " -o " + os.path.join(init.db_root,init.out_dir,uid + "_sminaord_docked.pdb")
#     cl = subprocess.Popen(reorder_cmd, shell=True, stdout=subprocess.PIPE)
#     cl.wait()
#     cl = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
#     cl.wait()
#     # read binding affinity prediction record as computed directly from crystal
#     ordered_file = os.path.join(init.out_dir, uid + "_sminaord.pdb")
#     with open(os.path.join(init.db_root,ordered_file)) as f: smina_out = f.readlines()
#     if smina_out[1].startswith("REMARK minimizedAffinity"):
#         ordered_aff = float(smina_out[1].split(" ")[-1])
#     else:
#         raise Exception("can not correctly parse file using hardwired template")
#     # read binding affinity prediction record as computed in a docked ligand
#     sminaord_docked_file = os.path.join(init.out_dir, uid + "_sminaord_docked.pdb")
#     with open(os.path.join(init.db_root,sminaord_docked_file)) as f: smina_out = f.readlines()
#     if smina_out[1].startswith("REMARK minimizedAffinity"): docked_aff = float(smina_out[1].split(" ")[-1])
#     else: raise Exception("can not correctly parse file using hardwired template")
#     # _returned docked file to a normal order
#     pr_lig = pr.parsePDB(os.path.join(init.db_root, lig_file))
#     pr_ordlig = pr.parsePDB(os.path.join(init.db_root, ordered_file))
#     pr_docked = pr.parsePDB(os.path.join(init.db_root, sminaord_docked_file))
#     lig_coord = pr_lig.select('noh').getCoords()
#     ordlig_coord = pr_ordlig.select('noh').getCoords()
#     docked_coords = pr_docked.select('noh').getCoordsets()
#     dist = np.sum((np.expand_dims(lig_coord,1) - np.expand_dims(ordlig_coord,0))**2,axis=2)
#     init_order = np.argmin(dist,axis=1)
#     assert len(np.unique(init_order)) == dist.shape[0], "no can not invert element order"
#     docked_coords = np.transpose(np.transpose(docked_coords,axes=[1,0,2])[init_order],[1,0,2])
#     docked_elem = pr_docked.select('noh').getElements()[init_order]
#     docked_atomnames = pr_docked.select('noh').getNames()[init_order]
#     docked_resnames = pr_docked.select('noh').getResnames()[init_order]
#
#     if not np.array_equal(np.char.lower(pr_lig.getElements()),np.char.lower(docked_elem)):
#         print "reodering broke",np.char.lower(pr_lig.getElements()),np.char.lower(docked_elem)
#     assert np.array_equal(np.char.lower(pr_lig.getElements()),np.char.lower(docked_elem)), "reordering broke"
#     pr_docked = pr.AtomGroup()
#     pr_docked.setCoords(docked_coords)
#     pr_docked.setElements(docked_elem)
#     pr_docked.setNames(docked_atomnames)
#     pr_docked.setResnames(docked_resnames)
#     docked_file = os.path.join(init.out_dir, uid + "_docked.pdb")
#     pr.writePDB(os.path.join(init.db_root,docked_file),pr_docked)
#     return [[uid, ordered_file, sminaord_docked_file, docked_file, ordered_aff, docked_aff]]
