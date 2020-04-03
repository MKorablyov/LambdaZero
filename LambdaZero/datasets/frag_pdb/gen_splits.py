"make samplers for the fragment database"
import os,sys,time
import numpy as np

class cfg:
    fragdb_path  = "/home/maksym/Projects/datasets/fragdb"
    fragdb_index = "ufrag_index.npz"
    gs = "/home/maksym/Projects/datasets/pdb/gs_05.npz"


if __name__ == "__main__":
    # load fragdb index file
    fragdb_index = np.load(os.path.join(cfg.fragdb_path, cfg.fragdb_index))
    pdb_ids = fragdb_index["pdb_ids"]
    lig_ids = fragdb_index["lig_ids"]
    len_fragdb = len(lig_ids)

    # load golden set of binding affinities for the pdb
    gs = np.load(cfg.gs)
    len_gs = len(gs["ligids"])
    gs_pdbids = gs["pdbids"].astype('U')
    gs_ligids = gs["ligids"].astype('U')

    gs_name_dict = dict([(gs_pdbids[i] + gs_ligids[i],i) for i in range(len_gs)])
    gs_pdbid_dict = dict([(gs_pdbids[i],i) for i in range(len_gs)])

    train_idxs = []
    test_idxs = []
    uMtrain_idxs = [] # micromolar or less binding affinities
    uMtest_idxs = []
    for idx in range(len_fragdb):
        name = pdb_ids[idx] + lig_ids[idx]
        if idx % 1000 ==1:
            print("done idx", idx)
        # regular train/test records split based on the pdb_id
        if pdb_ids[idx] in gs_pdbid_dict:
            gs_idx = gs_pdbid_dict[pdb_ids[idx]]
            if gs["train"][gs_idx]:
                train_idxs.append(idx)
            else:
                test_idxs.append(idx)

        # records with high affinity check exact match pdb_id + lig_id
        if name in gs_name_dict:
            gs_idx = gs_name_dict[name]
            if gs["log_affinities"][gs_idx] < np.log(10.0 ** -6):
                if gs["train"][gs_idx]:
                    uMtrain_idxs.append(idx)
                else:
                    uMtest_idxs.append(idx)

    # save indices
    np.save(os.path.join(cfg.fragdb_path, "train_idxs.npy"),np.asarray(train_idxs, dtype=np.int32))
    np.save(os.path.join(cfg.fragdb_path, "test_idxs.npy"), np.asarray(test_idxs, dtype=np.int32))
    np.save(os.path.join(cfg.fragdb_path,"uMtrain_idxs.npy"),np.asarray(uMtrain_idxs, dtype=np.int32))
    np.save(os.path.join(cfg.fragdb_path, "uMtest_idxs.npy"), np.asarray(uMtest_idxs, dtype=np.int32))
    print("saved records:", len(train_idxs), len(test_idxs),len(uMtrain_idxs),len(uMtest_idxs))

