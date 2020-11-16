import os,sys,time, random, string
import os.path as osp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
from sklearn.metrics import roc_curve, auc, roc_auc_score





# class GenMolFile_v1:
#     def __init__(self, outpath, num_conf):
#         self.outpath = outpath
#         self.num_conf = num_conf
#     def __call__(self, smi,mol_name):
#         mol = Chem.MolFromSmiles(smi)
#         Chem.SanitizeMol(mol)
#         mol_h = Chem.AddHs(mol)
#         AllChem.EmbedMultipleConfs(mol_h, numConfs=self.num_conf)
#         [AllChem.MMFFOptimizeMolecule(mol_h, confId=i) for i in range(self.num_conf)]
#         mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
#         # choose minimum energy conformer
#         mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy()
#                         for i in range(self.num_conf)])
#         mol = Chem.RemoveHs(mol_h)
#         # save file in .mol format
#         mol_file = os.path.join(self.outpath, mol_name + ".mol")
#         print(Chem.MolToMolBlock(mol, confId=int(mi)), file=open(mol_file, 'w+'))
#         return mol_file



# class GenMolFile_v1:
#     def __init__(self, outpath, num_conf):
#         self.outpath = outpath
#         self.num_conf = num_conf
#     def __call__(self, smi,mol_name):
#         mol = Chem.MolFromSmiles(smi)
#         Chem.SanitizeMol(mol)
#         mol_h = Chem.AddHs(mol)
#         AllChem.EmbedMultipleConfs(mol_h, numConfs=1)
#         [AllChem.MMFFOptimizeMolecule(mol_h, confId=i) for i in range(1)]
#         #mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
#         # choose minimum energy conformer
#         #mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy()
#         #                for i in range(self.num_conf)])
#         #mol = Chem.RemoveHs(mol_h)
#         # save file in .mol format
#         mol_file = os.path.join(self.outpath, mol_name + ".mol")
#         print(Chem.MolToMolBlock(mol, confId=0), file=open(mol_file, 'w+'))
#         return mol_file


class GenMolFile_v1:
    def __init__(self, mgltools, outpath, num_conf):
        self.mgltools = mgltools
        self.outpath = outpath
        self.num_conf = num_conf

    def __call__(self, smi,mol_name):
        mol = Chem.MolFromSmiles(smi)
        Chem.SanitizeMol(mol)
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol_h, numConfs=self.num_conf)
        [AllChem.MMFFOptimizeMolecule(mol_h, confId=i) for i in range(self.num_conf)]
        mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
        # choose minimum energy conformer
        mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy()
                        for i in range(self.num_conf)])
        lname = os.path.join(self.outpath, mol_name)
        #print(Chem.MolToMolBlock(mol_h, confId=int(mi)), file=open(lname + ".mol", 'w+'))
        #return lname + ".mol"
        print(Chem.MolToMolBlock(mol_h, confId=int(mi)), file=open(lname + ".sdf", 'w+'))
        os.system('babel -isdf {0} -omol2 {1}'.format(lname + ".sdf", lname + ".mol2"))
        prepare_ligand4 = osp.join(self.mgltools, "AutoDockTools/Utilities24/prepare_ligand4.py")
        os.system("pythonsh {} -l {} -o {}".format(prepare_ligand4,
                                                   lname + ".mol2",
                                                   lname + ".pdbqt"))
        return lname + ".pdbqt"



class DockVina_smi:
    def __init__(self, config):
        self.config = config
        # make vina command
        self.dock_cmd = "{} --receptor {} " \
                        "--center_x {} --center_y {} --center_z {} " \
                        "--size_x {} --size_y {} --size_z {} "
        self.dock_cmd = self.dock_cmd.format(config["vina_bin"], config["rec_file"], *config["bindsite"])
        self.dock_cmd += " --ligand {} --out {}"
        self.gen_molfile = config["gen_molfile"](**config["gen_molfile_par"])

    def dock(self, smi, mol_name=None):
        try:
            # generate random molecule name if needed
            if mol_name is None:
                mol_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
            #make mol file to dock
            dock_file = self.gen_molfile(smi, mol_name)
            #lig_file = osp.join(self.config["docksetup_dir"], "lig.pdb")
            sminaord_file = osp.join(self.config["outpath"], mol_name + "_sminaord.pdb")
            dock_cmd = self.dock_cmd.format(dock_file,sminaord_file)
            dock_cmd = dock_cmd + " " + self.config["dock_pars"]
            # dock
            cl = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
            cl.wait()
            # parse energy
            with open(os.path.join(sminaord_file)) as f: smina_out = f.readlines()
            if smina_out[1].startswith(
                    "REMARK VINA RESULT"):
                    #"REMARK minimizedAffinity"):
                dockscore = float(smina_out[1].split()[3])
                #dockscore = float(smina_out[1].split(" ")[-1]) # fixme
            else:
                raise Exception("can't correctly parse docking energy")
            print("docskscore", dockscore)
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
        "outpath":"/home/maksym/Datasets/seh/4jnc/docked",
        "vina_bin":
            "/home/maksym/Programs/vina/bin/vina",
            #"/home/maksym/Programs/smina/smina.static",
        "rec_file": "/home/maksym/Datasets/seh/4jnc/4jnc.nohet.aligned.pdbqt",
        "bindsite": [-13.4, 26.3, -13.3, 20.013, 16.3, 18.5],
        "docksetup_dir":"/home/maksym/Datasets/seh/4jnc",
        "dock_pars": "", #"--scoring vina", #"--exhaustiveness 8 --cpu 1",
        "gen_molfile": GenMolFile_v1,
        "gen_molfile_par": {
            "mgltools": "/home/maksym/Programs/mgltools_i86Linux2_1.5.6/MGLToolsPckgs/",
            "outpath":"/home/maksym/Datasets/seh/4jnc/docked",
            "num_conf":20,
        }}

    data = pd.read_csv("/home/maksym/Datasets/seh/seh_chembl.csv", sep=";")
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
    dockscore = pd.DataFrame({"dockscore":[dock_smi.dock(smi) for smi in smis]})
    binding = pd.concat([binding, dockscore],axis=1)
    binding.to_feather(os.path.join(config["outpath"], "seh.ftr"))
    binding = pd.read_feather(os.path.join(config["outpath"], "seh.ftr"))
    #binding = pd.read_feather("/home/maksym/Datasets/seh/4jnc/docked/seh_v1.ftr")
    binding = binding.dropna()

    fpr, tpr, _ = roc_curve(binding["Standard Value"] < 1, -binding["dockscore"])
    roc_auc = auc(fpr,tpr)
    # plt.plot(fpr,tpr)
    # plt.show()
    print(roc_auc)
    #print(roc_auc_score(binding["Standard Value"] < 1, -binding["dockscore"]))
    #plt.scatter(np.log(binding["Standard Value"]), binding["dockscore"])
    #plt.show()




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