import cv2

from argparse import Namespace
import os.path as osp
from rdkit.Chem import QED


import numpy as np
import torch
import argparse
import random
from rdkit.Chem import AllChem

from LambdaZero.examples.gflow.proxy_wrappers import CandidateWrapper
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def run(args, data_path):

    device = torch.device("cuda")
    proxy = CandidateWrapper(args.proxy)
    proxy.to(device)

    data = torch.load(data_path)

    print("FINALLY READ THE DATA FOLDER")
    all_children = data["mol_leaves"]
    mol_graphs = data["mol_graphs"]
    smiles = []

    if False:
        from true_parents import load_mdp, hashabledict
        test = "debug_131"
        bpath = osp.join(datasets_dir, f"fragdb/{test}.json")

        mdp, get_num_blocks = load_mdp(bpath)
        uniqids = [hashabledict(mdp.unique_id(x)) for x in all_children]

        new_smi = torch.load("/scratch/andrein/Datasets/Datasets/debug_extra/new_0.pk")["new"]
        for ix, new_mol in enumerate(new_smi):
            ngraph = mdp.get_nx_graph(new_mol, True)

            new_id = hashabledict(mdp.unique_id(new_mol))
            same_hash = np.where([new_id == x for x in uniqids])
            found = False

            for ix, (molg, (paths, parents, smiles)) in enumerate(mol_graphs):
                if mdp.graphs_are_isomorphic(molg, ngraph):
                    print("same", new_mol.smiles, smiles)
                    found = True
                    break
            if not found:
                print("NO SIMETRY !?!?!")
                import pdb; pdb.set_trace()
        exit()

    proxy_model = proxy.proxy
    all_infos = []
    count_smiles = dict()
    batch_size = 64
    num_paths = 0

    num_parents = []
    for ix, (molg, (paths, parents, smiles)) in enumerate(mol_graphs):
        count_smiles[list(smiles)[0]] = paths
        num_paths += paths
        num_parents.append(paths)

    num_atoms = [AllChem.MolFromSmiles(x).GetNumAtoms() for x in count_smiles.keys()]
    print(f"Avg size num atoms {np.mean(num_atoms)} | "
          f"max: {np.max(num_atoms)} | min: {np.min(num_atoms)} | median {np.median(num_atoms)}")
    print(f"Num Parents max {np.max(num_parents)} | mean {np.mean(num_parents)} "
          f"| median {np.median(num_parents)}"
          f"\nHistogram for number of parents:\n {np.histogram(num_parents)}")

    with torch.no_grad():
        for btch, i in enumerate(range(0, len(all_children), batch_size)):
            mols = all_children[i: i + batch_size]
            res_scores, infos = proxy(mols)

            mmm = [xmol.mol for xmol in mols]
            valsqed = [QED.qed(xm) for xm in mmm]
            valssynth = list(proxy.synth_net(mmm))
            for ix, (qed, synth) in enumerate(zip(valsqed, valssynth)):
                infos[ix]["qed"] = qed
                infos[ix]["synth"] = synth

            # Let's calculate proxy score for all molecules even if they are not good candidates
            proxy_scores = list(proxy_model(mols))
            for info, proxy_s, rscore in zip(infos, proxy_scores, res_scores):
                info["proxy"] = proxy_s
                info["rscore"] = rscore

            all_infos += infos
            if (btch + 1) % 100 == 0:
                print(f"Done {btch} / {len(all_children)//batch_size}")

    all_proxy_scores = [x["proxy"] for x in all_infos]
    all_qed_scores = [x["qed"] for x in all_infos]
    hist_bins = np.linspace(np.min(all_proxy_scores), np.quantile(all_proxy_scores, 0.9), 15)
    sorted_idxs = np.argsort(all_proxy_scores)[::-1]
    sorted_p = [all_proxy_scores[ix] for ix in sorted_idxs]

    candidates = [x.get("score") is not None for x in all_infos]
    num_good = np.sum(candidates)
    print(f"Good candidates: {num_good}/{len(candidates)} ({num_good/len(candidates)*100:.2f}%")

    print(f"Number of leaves: {len(all_children)} with {num_paths} number of paths to them & "
          f"{len(np.unique(all_proxy_scores))} unique scores from proxy")
    print(f"All Proxy mean {np.mean(all_proxy_scores)} | Min {np.min(all_proxy_scores)} "
          f" | max {np.max(all_proxy_scores)} | median {np.median(all_proxy_scores)} |"
          f" Top100: {np.mean(sorted_p[:100])}")
    print(f"All Proxy hist: \n {np.histogram(all_proxy_scores, bins=hist_bins)}")

    p_with_q = list(zip(sorted_p, [all_qed_scores[ix] for ix in sorted_idxs]))
    for topk in [10, 100, 1000, 5000]:
        print(f"Unique scores in top {topk} - {len(np.unique(sorted_p[:topk]))}")
    # for topk in [10, 100, 1000, 5000]:
    #     print(f"Unique scores/qed in top {topk} - {len(np.unique(p_with_q[:topk], axis=0))}")

    # ==============================================================================================
    # Choose sample of mols
    sort_proxy = np.argsort(all_proxy_scores)
    nump = len(all_proxy_scores)
    mol_samples_idx = np.concatenate([
        sort_proxy[:10].tolist(),  # top 10
        random.sample(sort_proxy[10:100].tolist(), k=20),  # random 10 from  top 10-100
        random.sample(sort_proxy[100:1000].tolist(), k=30),  # random 10 from  top 10-100
        random.sample(sort_proxy[int(0.3 * nump):int(0.7 * nump)].tolist(), k=40),  # random 10 avg
        random.sample(sort_proxy[-1000:].tolist(), k=10), # random 10 bottom 1000
    ])
    mol_samples = [all_children[ix] for ix in mol_samples_idx]
    # ==============================================================================================

    out_path = f"{data_path}_all.pk"
    print(f"Saving @ {out_path}")

    torch.save({
        "infos": all_infos,
        "count_smiles": count_smiles,
        "mol_samples": mol_samples,
        "mol_samples_idx": mol_samples_idx,
    }, out_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a reward model (brutal dock)')
    parser.add_argument('path', type=str)
    other_args = parser.parse_args()

    device = torch.device("cuda")
    args = Namespace(
        mdp_init=Namespace(repr_type="atom_graph"),
        proxy=Namespace(
            name="CandidateWrapper", default_score=16, device=device,
            proxy=Namespace(name="ProxyExample", device=device,
                            checkpoint="/scratch/andrein/Datasets/Datasets/best_model.pk")
        )
    )
    run(args, other_args.path)
