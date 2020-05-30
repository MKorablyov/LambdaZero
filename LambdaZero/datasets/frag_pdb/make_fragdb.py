import os,sys,time
from collections import Counter
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
import pandas as pd
#sys.path.append("../../")
from LambdaZero import chem

class cfg:
    #source_molfile = "/home/maksym/Datasets/pdb/old/lig_noHnoMiss.sdf"
    #source_molfile = "/home/maksym/Projects/datasets/pdb/example.sdf"
    source_molfile = "/home/maksym/Datasets/fragdb/lig_pdb_noh_miss_small.sdf"
    fragdb_path = "/home/maksym/Datasets/fragdb/fragdb"
    #fragspace_path = "/home/maksym/Desktop/temp_frag/fragspace"
    frags_generic = False
    block_cutoff = 100
    r_cutoff = 100


if __name__ == "__main__":
    fragdb = chem.FragDB()

    def fragdb_from_molfile(molfile, maxmol=10000000):
        mols = Chem.SDMolSupplier(molfile)
        passed_nmols = 0
        broken_nmols = 0
        for mol in mols:
            try:
                mol_frags = fragdb.add_mol(mol, frags_generic=cfg.frags_generic)
                if mol_frags is not None:
                    passed_nmols += 1
                else:
                    broken_nmols += 1
            except Exception as e:
                broken_nmols += 1
            print(passed_nmols, broken_nmols)
            if passed_nmols > maxmol:
                break
        print("molecules passed:", passed_nmols, "molecules broken", broken_nmols)

    def fragment_mols(molfile):
        mols = Chem.SDMolSupplier(molfile)
        passed_nmols = 0
        broken_nmols = 0
        for mol in mols:
            try:
                fragdb.get_mol_ufrag(mol, frags_generic=cfg.frags_generic)
                passed_nmols +=1
            except Exception as e:
                print("error:", e)
                broken_nmols += 1
            print("passed/broken", passed_nmols,"/", broken_nmols)
    fragdb_from_molfile(cfg.source_molfile)
    fragdb.save_state(cfg.fragdb_path)
    fragdb.load_state(cfg.fragdb_path, ufrag=False)

    fragdb.make_ufrags()
    block_names, block_smis, _, block_rs = fragdb.filter_ufrags(verbose=False,
                                                                block_cutoff=cfg.block_cutoff,
                                                                r_cutoff=cfg.r_cutoff)

    blocks = pd.DataFrame({"block_name":block_names,"block_smi": block_smis,"block_r":block_rs})
    blocks.to_json("/home/maksym/Datasets/fragdb/example_blocks.json")

    print(pd.read_json("/home/maksym/Datasets/fragdb/example_blocks.json"))


    #fragdb.draw_ufrag("/home/maksym/Datasets/fragdb/ufrag_img")
    #print(fragdb.ufrag_names, fragdb.ufrag_counts)
    #fragdb.save_state(cfg.fragdb_path)
    #fragdb.load_state(cfg.fragdb_path,ufrag=True)
    #fragment_mols(cfg.source_molfile)
