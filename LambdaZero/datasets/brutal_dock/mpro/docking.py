import os, sys
import random, string

#import subprocess
#import json
#import tempfile
#import prody as pr
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

# class DockVina:
#     def __init__(self,
#                  smina_bin,
#                  docksetup_dir,
#                  autobox_lig,
#                  out_dir):
#
#         if not os.path.exists(out_dir): os.makedirs(out_dir)
#         self.out_dir = out_dir
#
#         self.vina_cmd = "{} -r {} -l {} --autobox_ligand {} -o {}"
#
#
#     def dock(self, smi, mol_name=None):
#         if mol_name is None: mol_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
#
#     # return score, coord



if __name__ == "__main__":

    from LambdaZero import chem
    class cfg:
        #smina_bin = ""
        #docksetup_dir = ""
        #autobox_lig = ""
        #lig_file = ""

        rec_file = "/home/maksym/Programs/dock6/tutorials/2_mpro/6lze_rec.pdb"
        lig_file = "/home/maksym/Programs/dock6/tutorials/2_mpro_6lze/6lze_lig.pdb"
        num_conf = 25


    #mol = Chem.MolFromPDBFile(cfg.lig_file)
    #smi = Chem.MolToSmiles(mol)
    smi = "[O-]C(=O)[C@H](C[C@@H]1CCNC1=O)NC(=O)[C@H](CC2CCCCC2)NC(=O)c3[nH]c4ccccc4c3"

    mol = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(mol)
    #smi = Chem.MolToSmiles(mol)
    #print(smi)

    mol_h = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol_h,numConfs=cfg.num_conf, useBasicKnowledge=True)
    [AllChem.MMFFOptimizeMolecule(mol_h,confId=i) for i in range(cfg.num_conf)]
    mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant='MMFF94')
    # choose minimum energy conformer
    mi = np.argmin([AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=i).CalcEnergy() for i in range(cfg.num_conf)])
    mol = Chem.RemoveHs(mol_h)
    # save file in .pdb format
    pdb_file = "/home/maksym/Programs/dock6/tutorials/2_mpro_6lze/1_struct/from_smi.mol"
    #pdb_file = os.path.join(outpath,mol_name + ".pdb")
    print(Chem.MolToMolBlock(mol, confId=int(mi)), file=open(pdb_file, 'w+'))
    #mol = chem.build_mol(smi)["mol"][0]


#class Dock_init:
    #this_module = sys.modules[__name__]
    #a#rg_types = [str, str, str, str]
    #out_types = [str, str, str, str, float, float]
    #out_names = ["uid", "sminaord_file", "sminaorddocked_file", "docked_file", "ordered_affinity", "docked_affinity"]
    # def __init__(self,db_root,smina_path,dock_pars,out_H=False,out_dir="docked"):
    #     self.db_root = db_root
    #     self.smina_path = smina_path
    #     self.dock_pars = dock_pars
    #     self.out_dir = out_dir
    #     out_path = os.path.join(db_root,out_dir)
    #     if not os.path.exists(out_path): os.makedirs(out_path)
    #     self.this_module.dock_init = self

# def dock(uid,lig_file,rec_file,autobox_lig,init="dock_init"):
#     """
#
#     :param uid:
#     :param rec_file:
#     :param lig_file:
#     :param autobox_lig:
#     :param init:
#     :return:
#     """
#     init = eval(init)
#     smina_cmd = init.smina_path
#     smina_cmd += " -r " + os.path.join(init.db_root,rec_file)
#     smina_cmd += " -l " + os.path.join(init.db_root,lig_file)
#     smina_cmd += " --autobox_ligand " + os.path.join(init.db_root,autobox_lig) + " "
#     # make the ligand with the order of atoms shuffled how Vina wants it, but with heavy atoms being in the same place
#     reorder_cmd = smina_cmd + " --score_only" + " -o " + os.path.join(init.db_root, init.out_dir, uid + "_sminaord.pdb")
#     # make the actual docking command
#     dock_cmd = smina_cmd+init.dock_pars + " -o " + os.path.join(init.db_root,init.out_dir,uid + "_sminaord_docked.pdb")
#     cl = subprocess.Popen(reorder_cmd, shell=True, stdout=subprocess.PIPE)
#     cl.wait()
#     cl = subprocess.Popen(dock_cmd, shell=True, stdout=subprocess.PIPE)
#     cl.wait()
    # # read binding affinity prediction record as computed directly from crystal
    # ordered_file = os.path.join(init.out_dir, uid + "_sminaord.pdb")
    # with open(os.path.join(init.db_root,ordered_file)) as f: smina_out = f.readlines()
    # if smina_out[1].startswith("REMARK minimizedAffinity"):
    #     ordered_aff = float(smina_out[1].split(" ")[-1])
    # else:
    #     raise Exception("can not correctly parse file using hardwired template")
    # # read binding affinity prediction record as computed in a docked ligand
    # sminaord_docked_file = os.path.join(init.out_dir, uid + "_sminaord_docked.pdb")
    # with open(os.path.join(init.db_root,sminaord_docked_file)) as f: smina_out = f.readlines()
    # if smina_out[1].startswith("REMARK minimizedAffinity"): docked_aff = float(smina_out[1].split(" ")[-1])
    # else: raise Exception("can not correctly parse file using hardwired template")
    # # _returned docked file to a normal order
    # pr_lig = pr.parsePDB(os.path.join(init.db_root, lig_file))
    # pr_ordlig = pr.parsePDB(os.path.join(init.db_root, ordered_file))
    # pr_docked = pr.parsePDB(os.path.join(init.db_root, sminaord_docked_file))
    # lig_coord = pr_lig.select('noh').getCoords()
    # ordlig_coord = pr_ordlig.select('noh').getCoords()
    # docked_coords = pr_docked.select('noh').getCoordsets()
    # dist = np.sum((np.expand_dims(lig_coord,1) - np.expand_dims(ordlig_coord,0))**2,axis=2)
    # init_order = np.argmin(dist,axis=1)
    # assert len(np.unique(init_order)) == dist.shape[0], "no can not invert element order"
    # docked_coords = np.transpose(np.transpose(docked_coords,axes=[1,0,2])[init_order],[1,0,2])
    # docked_elem = pr_docked.select('noh').getElements()[init_order]
    # docked_atomnames = pr_docked.select('noh').getNames()[init_order]
    # docked_resnames = pr_docked.select('noh').getResnames()[init_order]
    #
    # if not np.array_equal(np.char.lower(pr_lig.getElements()),np.char.lower(docked_elem)):
    #     print "reodering broke",np.char.lower(pr_lig.getElements()),np.char.lower(docked_elem)
    # assert np.array_equal(np.char.lower(pr_lig.getElements()),np.char.lower(docked_elem)), "reordering broke"
    # pr_docked = pr.AtomGroup()
    # pr_docked.setCoords(docked_coords)
    # pr_docked.setElements(docked_elem)
    # pr_docked.setNames(docked_atomnames)
    # pr_docked.setResnames(docked_resnames)
    # docked_file = os.path.join(init.out_dir, uid + "_docked.pdb")
    # pr.writePDB(os.path.join(init.db_root,docked_file),pr_docked)
    # return [[uid, ordered_file, sminaord_docked_file, docked_file, ordered_aff, docked_aff]]


#
# Dock_init(db_root="/home/maksym/Desktop/super_dud_v8",
#           smina_path="/home/maksym/Programs/smina/smina.static",
#           dock_pars="--energy_range {} --exhaustiveness {} --autobox_add {}".format(100,1,5))
# dock(uid="101D_A_9_CBR",
#      rec_file="split_pdbs/100D_A_21_SPM/100D_A_21_SPM_bindsite.pdb", #"download_pdbs/101D.pdb",
#      lig_file="split_pdbs/100D_A_21_SPM/100D_A_21_SPM_ligand.pdb",
#      autobox_lig="split_pdbs/100D_A_21_SPM/100D_A_21_SPM_ligand.pdb")