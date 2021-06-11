import pandas as pd
from LambdaZero.environments import BlockMolEnvGraph_v1
from LambdaZero import chem
from rdkit import Chem
import glob
import torch

import json

def merge_new_data():

    # Read
    new_data_files = glob.glob(f"/home/andrein/scratch/tmp/new_data*")
    print("Reading from:", new_data_files)

    new_data = dict()
    for nf in new_data_files:
        newd = torch.load(nf)
        if isinstance(newd, dict):
            new_data.update(newd)
            continue

        for idp, data_point in enumerate(newd):
            if isinstance(data_point, tuple):
                if isinstance(data_point[0], dict):
                    new_data.update(data_point[0])
                else:
                    print("1", idp, data_point)
            else:
                if isinstance(data_point, dict):
                    new_data.update(data_point)
                else:
                    print("2", idp,data_point)


    torch.save(new_data, "/home/andrein/scratch/tmp/new_data_merged")


def fix_all_data():
    env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True})
    env.max_steps = 30

    dft = pd.read_pickle("data/trajectories/all_data.pk")  # type: pd.DataFrame

    bad_idxs = []
    for i, row in dft.iterrows():
        bad_index = None
        blocks = [env.molMDP.block_mols[x] for x in row["mol_data_blockidxs"]]
        try:
            mol, _ = chem.mol_from_frag(jun_bonds=row["mol_data_jbonds"], frags=blocks)
            smi = Chem.MolToSmiles(mol)
        except:
            bad_index = True
            continue
        if row["mol_data_smiles"] != smi or bad_index:
            if i in new_data:
                new_mol_data = new_data[i]
                for k, v in new_mol_data.items():
                    row[k] = v
            else:
                bad_idxs.append(i)
                print(i, "Not in new_data")

        if (i + 1) % 1000 == 0:
            print(i)
            break

    torch.save(bad_idxs, "/home/andrein/scratch/tmp/bad_idxs")
    dft.to_pickle("/home/andrein/scratch/tmp/all_data_fix.pk")


def check_mol(env, blockidxs, jbonds):
    blocks = [env.molMDP.block_mols[x] for x in blockidxs]
    works = True
    smi = None
    try:
        mol, _ = chem.mol_from_frag(jun_bonds=jbonds, frags=blocks)
        smi = Chem.MolToSmiles(mol)
    except:
        works = False

    return works, smi


def fix_db():
    from LambdaZero.contrib.oracle.oracle import PreDockingDB

    env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True})

    pathdb, all_dbs = PreDockingDB.get_last_db()
    print(pathdb)
    store = pd.HDFStore(pathdb, "r")
    db = store.select('df')

    new_data = torch.load("/home/andrein/scratch/tmp/new_data_merged")

    map_cl = dict({
        "mol_data_blockidxs": "blockidxs",
        "mol_data_slices": "slices",
        "mol_data_numblocks": "numblocks",
        "mol_data_jbonds": "jbonds",
        "mol_data_stems": "stems",
        "mol_data_smiles": "smiles",
    })

    nsmi_data = dict()
    for k, v in new_data.items():
        nsmi_data[v['mol_data_smiles']] = v

    dn = pd.DataFrame.from_dict(nsmi_data).transpose()
    dn = dn.rename(columns=map_cl)
    dn = dn.drop("smiles", axis=1)

    for k in dn.columns:
        dn[k] = dn[k].apply(lambda x: str(eval(str(x).replace("array", "list"))))

    # db.loc[db.index.isin(dn.index), ['numblocks']] = dn[['numblocks']]
    replaced = 0
    cnt_ = 0
    dn_indexes = dn.index

    flt = []
    for x in db.index: flt.append(x in dn_indexes)
    knames = db.index[flt]

    for cl in  ["blockidxs", "slices", "numblocks", "jbonds", "stems"]:
        db.loc[flt, cl] = dn.loc[knames, cl]

    print("REPLACED")
    # rnames = []
    # for rname in db.index.unique():
    #     if rname in dn_indexes:
    #         if len(db.loc[rname].shape) == 2:
    #             replaced += len(db.loc[rname])
    #         else:
    #             replaced += 1
    #
    #         # print(rname)
    #         for cl in ["blockidxs", "slices", "numblocks", "jbonds", "stems"]:
    #             # db.loc[rname, cl] = dn.loc[rname, cl]
    #             add_data[cl].append(dn.loc[rname, cl])
    #         rnames.append(rname)
    #
    #     cnt_ += 1
    #     if cnt_ % 100 == 0:
    #         print("Another", cnt_)

        # if 'C1CNCCN1' == rname:
        #     print(db.loc['C1CNCCN1'])
    # import pdb; pdb.set_trace()
    replaced = 0
    # for smi in db.index.unique():
    #     if smi in nsmi_data:
    #         if len(db.loc[smi].shape) == 2:
    #             dim2 = True
    #         else:
    #             dim2 = False
    #
    #         replaced += len(db.loc[smi]) if dim2 else 1
    #         for k, v in map_cl.items():
    #             if dim2:
    #                 db.loc[smi, v] = [str(nsmi_data[smi][k])] * len(db.loc[smi])
    #             else:
    #                 db.loc[smi, v] = str(nsmi_data[smi][k])

    print("Replaced", replaced)
    bad_index = []
    cnt = 0
    for index, row in db.iterrows():
        try:
            blckidxs = json.loads(row["blockidxs"])
            jbonds = json.loads(row["jbonds"])
        except:
            import pdb; pdb.set_trace()
        good, new_smi = check_mol(env, blckidxs, jbonds)

        if len(jbonds) != len(blckidxs) - 1 or new_smi != index:
            good = False

        if not good:
            bad_index.append(index)
        cnt += 1
        if cnt % 1000 == 0:
            print("Done", cnt, len(bad_index))

    print("bad cnt", len(bad_index))
    import pdb; pdb.set_trace()

    new_db = db.drop(bad_index)

    new_db_path = "/home/andrein/scratch/tmp/new_db"
    print(f"Write new db @ {new_db_path} ... ")
    new_db.to_hdf(new_db_path, 'df', mode='w')
    print(f"Writen!")


    torch.save(bad_index, "/home/andrein/scratch/tmp/bad_db_indexs")

def test2():
    import json
    import pandas as pd
    path = "/home/andrei/dock_db_1619111711tp_2021_04_22_13h.h5"

    columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
    store = pd.HDFStore(path)
    df = store.select('df')
    df.dockscore = df.dockscore.astype(
        "float64")  # Pandas has problem with calculating some stuff on float16
    for cl_mame in columns[2:]:
        df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)

    df.groupby("numblocks").plot(kind="box")
    df.dockscore.isn

def test_again():
    from LambdaZero.contrib.oracle.oracle import PreDockingDB

    store = pd.HDFStore("/home/andrein/scratch/tmp/new_db", "r")
    db = store.select('df')
    env = BlockMolEnvGraph_v1({"random_steps": 1, "allow_removal": True})

    bad_index = []
    cnt = 0
    for index, row in db.iterrows():
        try:
            blckidxs = json.loads(row["blockidxs"])
            jbonds = json.loads(row["jbonds"])
        except:
            import pdb; pdb.set_trace()
        good, new_smi = check_mol(env, blckidxs, jbonds)

        if len(jbonds) != len(blckidxs) - 1 or new_smi != index:
            good = False

        if not good:
            bad_index.append(index)
        cnt += 1
        if cnt % 1000 == 0:
            print("Done", cnt, len(bad_index))


    print(len(bad_index))
    k = input("SAVE New?")
    if k == "y":
        new_db_path = PreDockingDB.get_new_db_name()
        print(f"Write new db @ {new_db_path} ... ")
        db.to_hdf(new_db_path, 'df', mode='w')
        print(f"Writen!")


if __name__ == "__main__":
    # merge_new_data()
    # fix_db()
    test_again()