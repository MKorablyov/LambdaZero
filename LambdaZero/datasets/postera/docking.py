import os,time, random, string
import os.path as osp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

def gen_molfile(smi, mol_name, outpath, num_conf):
    # generate num_conf 3D conformers from smiles
    mol = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(mol)
    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol_h,numConfs=num_conf)
    [AllChem.MMFFOptimizeMolecule(mol_h,confId=i) for i in range(num_conf)]
    mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
    # choose minimum energy conformer
    mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(num_conf)])
    mol = Chem.RemoveHs(mol_h)
    # save file in .mol format
    mol_file = os.path.join(outpath,mol_name + ".mol")
    print(Chem.MolToMolBlock(mol,confId=int(mi)),file=open(mol_file,'w+'))
    return mol_file


class DockVina_smi:
    def __init__(self,
                 outpath,           # directory where to write stuff
                 smina_bin,         # smina binary
                 docksetup_dir,     # folder with lig.pdb and rec.pdb
                 molgen_conf=20):
        self.outpath = outpath
        self.smina_bin = smina_bin
        self.docksetup_dir = docksetup_dir
        self.molgen_conf = molgen_conf

    def dock(self, smi, mol_name=None):
        try:
            # generate random molecule name if needed
            if mol_name is None:
                mol_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))

            #make mol file to dock
            dock_file = gen_molfile(smi, mol_name, self.outpath, self.molgen_conf)
            # make vina command
            # todo: dock_pars
            rec_file = osp.join(self.docksetup_dir, "rec.pdb")
            lig_file = osp.join(self.docksetup_dir, "lig.pdb")
            sminaord_file = osp.join(self.outpath, mol_name + "_sminaord.pdb")
            dock_cmd = "{} -r {} -l {} --autobox_ligand {} -o {}".\
                format(self.smina_bin,rec_file,dock_file,lig_file,sminaord_file)
            # dock
            cl = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
            cl.wait()
            # parse energy
            with open(os.path.join(sminaord_file)) as f: smina_out = f.readlines()
            if smina_out[1].startswith("REMARK minimizedAffinity"):
                dockscore = float(smina_out[1].split(" ")[-1])
            else:
                raise Exception("can't correctly parse docking energy")
            return dockscore
        except Exception as e:
            return None



if __name__ == "__main__":
    data = pd.read_csv("/home/maksym/Datasets/seh/seh_chembl.csv", sep=";")
    binding = data[["Smiles", "Standard Value", "Ligand Efficiency BEI", "Ligand Efficiency SEI"]].copy()
    binding["Standard Value"] = pd.to_numeric(binding["Standard Value"], errors="coerce")
    binding["Ligand Efficiency BEI"] = pd.to_numeric(binding["Ligand Efficiency BEI"], errors="coerce")
    binding["Ligand Efficiency SEI"] = pd.to_numeric(binding["Ligand Efficiency SEI"], errors="coerce")
    binding = binding.dropna()
    smis = binding["Smiles"].to_numpy()
    # plt.scatter(binding["Ligand Efficiency SEI"], binding["Ligand Efficiency BEI"])
    # plt.show()

    class cfg:
        dock_out = "/home/maksym/Datasets/seh/4jnc/docked"
        smina_bin = "/home/maksym/Programs/smina/smina.static"
        docksetup_dir = "/home/maksym/Datasets/seh/4jnc"


    dock_smi = DockVina_smi(cfg.dock_out,cfg.smina_bin,cfg.docksetup_dir)


    dockscore = pd.DataFrame({"dockscore":[dock_smi.dock(smi) for smi in smis[:50]]})
    binding = pd.concat([binding, dockscore],axis=1)

    binding = binding.dropna()
    plt.scatter(np.log(binding["Standard Value"]), binding["dockscore"])
    plt.show()





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