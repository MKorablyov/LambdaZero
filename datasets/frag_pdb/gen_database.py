import os,sys,time,shutil
import numpy as np
from rdkit import Chem
from collections import Counter
sys.path.append("../../")
from affinity_torch.py_tools import fragdb
from affinity_torch.py_tools import multithread

class cfg:
    # PDB ligand dataset
    #root = "/mas/u/mkkr/datasets/"
    root = "/home/maksym/Projects/datasets/"

    pdblig_folder = os.path.join(root,"pdb/lig")
    pdblig_index = os.path.join(root,"pdb/lig_index.npy")

    # bindsite dataset
    dud_root = os.path.join(root,"super_dud_v8/")
    dud_index = os.path.join(root,"super_dud_v8/npy_records/db_index.npy")

    # fragdb (this) dataset
    fragdb_path = os.path.join(root,"fragdb")
    fragdb_state = os.path.join(root,"fragdb/fragdb") # todo temp
    frags_generic = False

    # general
    num_threads = 25

    # runtime shared variables
    DUD_UIDS = None
    FRAGDB = None


def process_fragdb_sample(lig_name,cfg=cfg):

    pdb_id, lig_id, chain, res, _, _ = lig_name.split("_")
    dud_uid = pdb_id + "_" + chain + "_" + res + "_" + lig_id
    assert dud_uid in cfg.DUD_UIDS, "receptor would be missing"

    # load ligand and break into groups
    molfile = os.path.join(cfg.pdblig_folder,lig_name + ".sdf")
    mol = Chem.SDMolSupplier(molfile)[0]
    frag_elems, frag_coords, jun_bonds, ufrag_idxs = cfg.FRAGDB.get_mol_ufrag(mol,frags_generic=cfg.frags_generic)

    #  convert to the desired data types before saving
    frag_elems = [np.asarray(frag_elem,dtype=np.float32) for frag_elem in frag_elems]
    frag_coords = [np.asarray(frag_coord, dtype=np.float32) for frag_coord in frag_coords]
    jun_bonds = np.asarray(jun_bonds,dtype=np.int32)
    ufrag_idxs = np.asarray(ufrag_idxs,dtype=np.int32)

    # save binding site and ligand
    np.savez(os.path.join(cfg.fragdb_path, "lig_frag", lig_name),
             frag_elems=frag_elems, frag_coords=frag_coords, jun_bonds=jun_bonds, ufrag_idxs=ufrag_idxs)
    shutil.copyfile(os.path.join(cfg.dud_root,"npy_records/rec_elem", dud_uid + ".npy"),
                    os.path.join(cfg.fragdb_path, "rec_elem", lig_name + ".npy"))
    shutil.copyfile(os.path.join(cfg.dud_root, "npy_records/rec_coord", dud_uid + ".npy"),
                    os.path.join(cfg.fragdb_path, "rec_coord", lig_name + ".npy"))
    return pdb_id, lig_id, lig_name, ufrag_idxs

if __name__ == "__main__":
    # make output directories
    os.makedirs(os.path.join(cfg.fragdb_path, "lig_frag"))
    os.makedirs(os.path.join(cfg.fragdb_path, "rec_elem"))
    os.makedirs(os.path.join(cfg.fragdb_path, "rec_coord"))

    # load config files
    lig_index = np.load(cfg.pdblig_index)#[:2000]
    dud_index = np.load(cfg.dud_index)
    cfg.DUD_UIDS = dud_index[:,0]

    # load fragment database
    cfg.FRAGDB = fragdb.FragDB()
    cfg.FRAGDB.load_state(cfg.fragdb_state, ufrag=True)
    fargs = [{"lig_name": lig_name} for lig_name in lig_index]
    passed_args, func_out, errors = multithread.run_multithread(process_fragdb_sample, fargs, cfg.num_threads)

    # save index of fragments
    # [lig_name, frag_number, frag_idx]  -> [lig_name, frag_number, prob]
    names = []
    pdb_ids = []
    lig_ids = []
    ufrag_nums = [] # [0,1,2,3,4] [ 2x num frags] for each molecule
    ufrag_idxs = []

    for i in range(len(func_out)):
        pdb_id, lig_id, name, ufrag_idx = func_out[i]
        for j in range(len(ufrag_idx)):
            names.append(name)
            pdb_ids.append(pdb_id)
            lig_ids.append(lig_id)
            ufrag_nums.append(j)
            ufrag_idxs.append(ufrag_idx[j])
    ufrag_count = Counter(ufrag_idxs)
    ufrag_probs = np.asarray([1. / ufrag_count[idx] for idx in ufrag_idxs],dtype=np.float32)
    ufrag_probs = ufrag_probs / ufrag_probs.mean()
    np.savez(os.path.join(cfg.fragdb_path,"ufrag_index.npz"),
             names=np.asarray(names),pdb_ids=np.asarray(pdb_ids),lig_ids=np.asarray(lig_ids),
             ufrag_nums=np.asarray(ufrag_nums, dtype=np.int32),
             ufrag_probs=np.asarray(ufrag_probs, dtype=np.float32))
    ufrag_index = np.load(os.path.join(cfg.fragdb_path, "ufrag_index.npz"))

    # done; final messages
    print(errors)
    print("prepared database pass/fail", len(func_out), len(errors))

    # num_pass = 0
    # num_fail = 0
    # for lig_name in lig_index:
    #     try:
    #         process_fragdb_sample(lig_name)
    #         num_pass +=1
    #     except Exception as e:
    #         num_fail +=1
    #         print(e)