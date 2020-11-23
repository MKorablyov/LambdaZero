import argparse
import os
import numpy as np
import pandas as pd

import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

root = os.environ['SLURM_TMPDIR']
datasets_dir = os.path.join(root, "Datasets")
programs_dir = os.path.join(root, "Programs")


class GenMolFile_zinc20:
    def __init__(self, mgltools, outpath, num_conf):
        self.prepare_ligand4 = os.path.join(mgltools, "AutoDockTools/Utilities24/prepare_ligand4.py")
        self.outpath = outpath
        self.num_conf = num_conf

        os.makedirs(os.path.join(outpath, "sdf"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "mol2"), exist_ok=True)
        os.makedirs(os.path.join(outpath, "pdbqt"), exist_ok=True)

    def __call__(self, smi, int_zinc_id):
        sdf_file = os.path.join(self.outpath, "sdf", f"{int_zinc_id}.sdf")
        mol2_file = os.path.join(self.outpath, "mol2", f"{int_zinc_id}.mol2")
        pdbqt_file = os.path.join(self.outpath, "pdbqt", f"{int_zinc_id}.pdbqt")

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
        return pdbqt_file


class DockVina_smi_zinc20:
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

        os.makedirs(os.path.join(self.outpath, "docked"), exist_ok=True)

    def dock(self, smi, int_zinc_id):
        dockscore = float('nan')
        status = "ok"
        try:
            docked_file = os.path.join(self.outpath, "docked", f"{int_zinc_id}.pdb")
            input_file = self.gen_molfile(smi, int_zinc_id)
            # complete docking query
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
        except Exception as e:
            status = str(e)

        return int_zinc_id, smi, dockscore, status


config = {
        "outpath": os.path.join(datasets_dir, "zinc20"),
        "vina_bin": os.path.join(programs_dir, "vina/bin/vina"),
        "rec_file": os.path.join(datasets_dir, "seh/4jnc/4jnc.nohet.aligned.pdbqt"),
        "bindsite": [-13.4, 26.3, -13.3, 20.013, 16.3, 18.5],
        "docksetup_dir": os.path.join(datasets_dir, "seh/4jnc"),
        "dock_pars": "",
        "gen_molfile": GenMolFile_zinc20,
        "gen_molfile_par": {
            "mgltools": os.path.join(programs_dir, "mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs"),
            "outpath": os.path.join(datasets_dir, "zinc20"),
            "num_conf": 20,
        }}

# Only small chunk of it is called, but file itself is relatively big. No point in copying it to SLURM_TMPDIR
dataset_path = "/home/mkkr/projects/rrg-bengioy-ad/mkkr/Datasets/zinc20/zinc20_filtered.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, help='Number of molecules in a batch')
    parser.add_argument('batch_idx', type=int, help='Index of a batch')
    args = parser.parse_args()
    batch_size, batch_idx = args.batch_size, args.batch_idx

    col_names = pd.read_csv(dataset_path, nrows=0).columns.tolist()                                         # smiles, int_zinc_id
    data = pd.read_csv(dataset_path, names=col_names, nrows=batch_size, skiprows=1+batch_idx*batch_size)    # additional 1 skips column names

    dock_smi = DockVina_smi_zinc20(config)

    docking_results = [dock_smi.dock(smi, idx) for smi, idx in zip(data['smiles'], data['int_zinc_id'])]
    docking_results = pd.DataFrame(docking_results, columns=['int_zinc_id', 'smiles', 'dockscore', 'status'])
    docking_results.to_csv(os.path.join(datasets_dir, "zinc20", f"subset_{batch_idx}.csv"), index=False)
