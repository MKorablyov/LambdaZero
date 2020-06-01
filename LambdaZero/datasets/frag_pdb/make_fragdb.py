import os,sys,time
import rdkit
from rdkit import Chem
import pandas as pd
from rdkit.Chem import AllChem
sys.path.append("../../")

from LambdaZero import chem

class cfg:
    source_molfile = "/home/nova/Downloads/zinc15_subset.feather"
    fragdb_path = "/home/nova/LambdaZero/datasets/fragdb/fragdb"
    frags_generic = False
    block_cutoff1 = 55
    block_cutoff2 = 105
    block_cutoff3 = 210
    block_cutoff4 = 330
    r_cutoff = 2
    datatype = "feather"


if __name__ == "__main__":
    fragdb = chem.FragDB()

    def build_mol(smiles=None, num_conf=1):
        mol = Chem.MolFromSmiles(smiles)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=num_conf)
        AllChem.ComputeGasteigerCharges(mol)
        mol = Chem.RemoveHs(mol)

        return mol

    def fragdb_from_molfile(molfile, smiles, f):
        passed_nmols = 0
        broken_nmols = 0
        if smiles:
            try:
                mol = build_mol(molfile)
                mol_frags = fragdb.add_mol(mol, frags_generic=cfg.frags_generic)
                if mol_frags is not None:
                    passed_nmols += 1
                else:
                    broken_nmols += 1
            except Exception as e:
                f.write(molfile)
                broken_nmols += 1
        else:
            mols = Chem.SDMolSupplier(molfile)
            for mol in mols:
                try:
                    mol_frags = fragdb.add_mol(mol, frags_generic=cfg.frags_generic)
                    if mol_frags is not None:
                        passed_nmols += 1
                    else:
                        broken_nmols += 1
                    print("molecules passed:", passed_nmols, "molecules broken", broken_nmols)
                except Exception as e:
                    broken_nmols += 1

        return passed_nmols, broken_nmols

    num_atoms = []

    if cfg.datatype == "feather":
        feather = pd.read_feather(cfg.source_molfile)
        running_passed_nmols = 0
        running_broken_nmols = 0
        i = 0
        f = open("broken_smiles.txt", "w")
        for index, row in feather.iterrows():
            passed_nmols, broken_nmols = fragdb_from_molfile(row['smiles'], True, f)
            running_passed_nmols += passed_nmols
            running_broken_nmols += broken_nmols
            print("molecules passed:", running_passed_nmols, "molecules broken", running_broken_nmols)
        f.close()
    else:
        fragdb_from_molfile(cfg.source_molfile, False)

    fragdb.save_state(cfg.fragdb_path)
    fragdb.load_state(cfg.fragdb_path, ufrag=False)

    fragdb.make_ufrags()

    block_names1, block_smis1, _, block_rs1 = fragdb.filter_ufrags(verbose=False,
                                                                    block_cutoff=cfg.block_cutoff1,
                                                                    r_cutoff=cfg.r_cutoff)

    blocks = pd.DataFrame({"block_name":block_names1,"block_smi": block_smis1,"block_r":block_rs1})
    blocks.to_json("/home/nova/vocabs/zinc15_blocks_55_wH.json")

    block_names2, block_smis2, _, block_rs2 = fragdb.filter_ufrags(verbose=False,
                                                                block_cutoff=cfg.block_cutoff2,
                                                                r_cutoff=cfg.r_cutoff)

    blocks = pd.DataFrame({"block_name": block_names2, "block_smi": block_smis2, "block_r": block_rs2})
    blocks.to_json("/home/nova/vocabs/zinc15_blocks_105_wH.json")

    block_names3, block_smis3, _, block_rs3 = fragdb.filter_ufrags(verbose=False,
                                                                block_cutoff=cfg.block_cutoff3,
                                                                r_cutoff=cfg.r_cutoff)

    blocks = pd.DataFrame({"block_name": block_names3, "block_smi": block_smis3, "block_r": block_rs3})
    blocks.to_json("/home/nova/vocabs/zinc15_blocks_210_wH.json")

    block_names4, block_smis4, _, block_rs4 = fragdb.filter_ufrags(verbose=False,
                                                                block_cutoff=cfg.block_cutoff4,
                                                                r_cutoff=cfg.r_cutoff)

    blocks = pd.DataFrame({"block_name": block_names4, "block_smi": block_smis4, "block_r": block_rs4})
    blocks.to_json("/home/nova/vocabs/zinc15_blocks_330_wH.json")
