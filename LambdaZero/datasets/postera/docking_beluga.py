import os
import random
import string
import numpy as np
import pandas as pd

import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import roc_curve, auc

root = os.environ['SLURM_TMPDIR']
datasets_dir = os.path.join(root, "Datasets")
programs_dir = os.path.join(root, "Programs")


class GenMolFile_v1:
    def __init__(self, mgltools, outpath, num_conf):
        self.prepare_ligand4 = os.path.join(mgltools, "AutoDockTools/Utilities24/prepare_ligand4.py")
        self.outpath = outpath
        self.num_conf = num_conf

        os.makedirs(os.path.join(outpath, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "mol2"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "pdbqt"), exist_ok=True)

    def obabel(self, sdf_file, mol2_file):
        os.system(f"obabel -isdf {sdf_file} -omol2 -O {mol2_file}")

    def pythonsh(self, mol2_file, pdbqt_file):
        os.system(f"pythonsh {self.prepare_ligand4} -l {mol2_file} -o {pdbqt_file}")

    def __call__(self, smi, mol_name):
        sdf_file = os.path.join(self.outpath, "sdf", f"{mol_name}.sdf")
        mol2_file = os.path.join(self.outpath, "mol2", f"{mol_name}.mol2")
        pdbqt_file = os.path.join(self.outpath, "pdbqt", f"{mol_name}.pdbqt")

        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol_h, numConfs=self.num_conf)
        [AllChem.MMFFOptimizeMolecule(mol_h, confId=i) for i in range(self.num_conf)]
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
        # choose minimum energy conformer
        mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(self.num_conf)])
        print(Chem.MolToMolBlock(mol_h, confId=int(mi)), file=open(sdf_file, 'w+'))
        os.system(f"obabel -isdf {sdf_file} -omol2 -O {mol2_file}")
        os.system(f"pythonsh {self.prepare_ligand4} -l {mol2_file} -o {pdbqt_file}")
        # self.obabel(sdf_file, mol2_file)
        # self.pythonsh(mol2_file, pdbqt_file)
        return pdbqt_file


class DockVina_smi:
    def __init__(self, config):
        self.gen_molfile = config["gen_molfile"](**config["gen_molfile_par"])
        self.outpath = config["outpath"]
        self.dock_pars = config["dock_pars"]
        # make vina command
        self.dock_cmd = "{} --receptor {} " \
                        "--center_x {} --center_y {} --center_z {} " \
                        "--size_x {} --size_y {} --size_z {} "
        self.dock_cmd = self.dock_cmd.format(config["vina_bin"], config["rec_file"], *config["bindsite"])
        self.dock_cmd += " --ligand {} --out {}"

    def dock(self, smi, mol_name=None):
        try:
            print(smi)
            # generate random molecule name if needed
            mol_name = mol_name or ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
            # make mol file to dock
            input_file = self.gen_molfile(smi, mol_name)
            docked_file = os.path.join(self.outpath, f"{mol_name}.pdb")
            dock_cmd = self.dock_cmd.format(input_file, docked_file)
            dock_cmd = dock_cmd + " " + self.dock_pars
            # dock
            cl = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
            cl.wait()
            # parse energy
            with open(docked_file) as f:
                smina_out = f.readlines()
            if smina_out[1].startswith("REMARK VINA RESULT"):
                dockscore = float(smina_out[1].split()[3])
            else:
                raise Exception("Can't parse docking energy")
            return dockscore
        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    # smina no hydrogens + relaxation: 0.86
    # vina no hydrogens 0.83
    # vina hydorgens 0.86
    # vina hydrogens + relaxation 0.856
    # vina + hydrogens + relaxation + Masha's script 0.864
    # vina + hydrogens + relaxation + Masha's script + bounding box + masha's pdbqt 0.847
    # vina + hydrogens + relaxation + Masha's script + bounding box + masha's pdbqt + original vina binary 0.9
    # vina + hydrogens + Masha's script + bounding box + masha's pdbqt + original vina binary 0.9
    # smina + hydrogens 0.819
    # Maksym repeat Maria's docking setup: 0.910
    # Maria's vina: 0.92

    # todo: strip salts

    config = {
        "outpath": os.path.join(datasets_dir, "seh/4jnc/docked"),
        "vina_bin": os.path.join(programs_dir, "vina/bin/vina"),
        "rec_file": os.path.join(datasets_dir, "seh/4jnc/4jnc.nohet.aligned.pdbqt"),
        "bindsite": [-13.4, 26.3, -13.3, 20.013, 16.3, 18.5],
        "docksetup_dir": os.path.join(datasets_dir, "seh/4jnc"),
        "dock_pars": "",
        "gen_molfile": GenMolFile_v1,
        "gen_molfile_par": {
            "mgltools": os.path.join(programs_dir, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs"),
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

    dock_smi = DockVina_smi(config)
    dockscore = pd.DataFrame({"dockscore": [dock_smi.dock(smi) for smi in smis]})
    binding = pd.concat([binding, dockscore], axis=1)
    binding.to_feather(os.path.join(config["outpath"], "seh.feather"))
    binding = pd.read_feather(os.path.join(config["outpath"], "seh.feather"))
    binding = binding.dropna()

    fpr, tpr, _ = roc_curve(binding["Standard Value"] < 1, -binding["dockscore"])
    roc_auc = auc(fpr, tpr)
    print(roc_auc, file=open(os.path.join(root, "roc_auc.txt"), 'w+'))
