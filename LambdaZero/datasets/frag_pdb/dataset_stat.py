"input for the fragdb"
import os,sys,time
from collections import Counter
import numpy as np
sys.path.append("../../")
from rdkit import Chem # todo: strange import problem
from affinity_torch.python_tools import fragdb
import torch
import affinity_torch as af
from affinity_torch.python_tools import chem
from matplotlib import pyplot as plt

class cfg:
    # dataset parameters
    root = "/home/maksym/Projects/datasets"
    db_path = os.path.join(root,"fragdb")
    ufrag_index = "ufrag_index.npz"

    # data preparation
    fragdb_state = os.path.join(root, "fragdb/fragdb")
    # dataset
    dataset_frag ={"db_path": db_path, "db_index": "ufrag_index.npz", "rotate": True}
    dataset_img = {"datasetFrag":dataset_frag,
               #"rootImgGen": {"grid_pix": 40, "pix_size": 0.45, "sigmas": [0.5, 0.75, 1.0], "input_d": 3},
               "rootImgGen":None,
               "fragImgGen": None,
               #"#recImgGen":{"grid_pix": 40, "pix_size": 0.45, "sigmas": [0.5, 0.75, 1.0], "input_d": 2},"cuda":True}
               "recImgGen":None}
    train_idxs = "train_idxs.npy"
    test_idxs = "test_idxs.npy"
    uM_test_idxs = "uMtest_idxs.npy"
    fragdb_path = "/home/maksym/Projects/datasets/fragdb/fragdb"

# test_idxs = np.load(os.path.join(cfg.db_path, cfg.test_idxs))
# test_sampler = af.inputs.Sampler(test_idxs, probs=ufrag_probs[test_idxs],num_samples=100000)

def make_conn_table(cfg):
    "plot connection table of the fragments in the database"
    train_idxs = np.load(os.path.join(cfg.db_path, cfg.train_idxs))
    ufrag_probs = np.load(os.path.join(cfg.db_path, cfg.ufrag_index))["ufrag_probs"]
    dataset = af.inputs.DatasetFrag(**cfg.dataset_frag)
    train_sampler = af.inputs.Sampler(train_idxs, probs=ufrag_probs[train_idxs], num_samples=100000)
    train_q = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, sampler=train_sampler)

    # compute conneciton prob table:
    frag_bar = np.zeros(105)
    root_bar = np.zeros(105)
    conn_table = np.zeros([105,105])

    for bnum,sample in enumerate(train_q):
        frag_bar[sample["ufrag_idx"]] +=1
        root_bar[sample["root_idx"]] += 1
        conn_table[sample["ufrag_idx"]][sample["root_idx"]] +=1
        if bnum % 100 ==1: print(bnum)
        if bnum > 10000: break

    #make images
    out_path = os.path.join(cfg.root,cfg.db_path,"dataset_stat")
    if not os.path.exists(out_path): os.makedirs(out_path)

    fig = plt.figure(dpi=200)
    plt.subplot(2,1,1)
    plt.imshow([frag_bar])
    plt.colorbar()
    plt.title("leaf fragment count")
    plt.subplot(2,1,2)
    plt.imshow([root_bar])
    plt.colorbar()
    plt.title("root fragment count")
    plt.savefig(os.path.join(out_path,"fragment_count_oversampled"))
    plt.close()

    fig = plt.figure(dpi=200)
    plt.imshow(conn_table)
    plt.xlabel("root fragment")
    plt.ylabel("leaf fragment")
    plt.title("fragment connection table")
    plt.colorbar()
    plt.savefig(os.path.join(out_path,"fragment_conn_table"))
    print("density:",np.asarray(conn_table > 0,dtype=np.float32).mean())


def rdkit_validity():
    from affinity_torch.python_tools import fragdb
    fragdb = fragdb.FragDB()
    fragdb.load_state(cfg.fragdb_path, ufrag=True)

    #r_groups = [np.asarray(r_group > 0,dtype=np.float32) for r_group in fragdb.frag_r]
    #mean_r = np.mean([np.max(r_group * np.array([[1,2,3,4]]),axis=1).sum() for r_group in r_groups])
    #print("mean r groups/fragment ", mean_r)

    # compute the expansion coeffient of each fragment
    sel_frag_names = [name.split("_")[0] for name in fragdb.ufrag_names]
    sel_frag_idx = [fragdb.frag_names[name] for name in sel_frag_names]
    print("mean number of r-groups:", np.mean(list(Counter(sel_frag_idx).values())))

    # check rdkit validity of each pair
    # sel_frag_r = [int(name.split("_")[1]) for name in fragdb.ufrag_names]
    # passed = 0
    # failed = 0
    # num_frag = len(sel_frag_names)
    # for i in range(num_frag):
    #     for j in range(num_frag):
    #         jun_bonds = np.asarray([0,1, sel_frag_r[i], sel_frag_r[j]])[None,:]
    #
    #         try:
    #             chem.build_mol([sel_frag_names[i],sel_frag_names[j]],jun_bonds,optimize=True)
    #             passed+=1
    #         except Exception as e:
    #             failed+=1
    #         print("pass/fail",(passed,failed)) # (8721, 2304)

    mean_natm = np.asarray([len(fragdb.frag_elem[fragdb.frag_names[name]]) for name in sel_frag_names],
                           dtype=np.float).mean()
    print("mean number of atoms", mean_natm)\
    # 12 steps * 5 atoms
    # (104 * 0.75 * 0.5)^12 = 39^12 = 1.23 * 10.19

if __name__ == "__main__":
    #make_conn_table(cfg)
    rdkit_validity()