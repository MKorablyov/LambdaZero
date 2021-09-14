# Test parents
import cv2
import os.path as osp
import torch
import numpy as np
import pandas as pd
import os
import time
import sys
import argparse
from collections import Counter

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended
from LambdaZero.utils import get_external_dirs


datasets_dir, programs_dir, summaries_dir = get_external_dirs()


class hashabledict(Counter):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def build_new_mdp():
    """ FIX fragdb with corrected blocks from 105 """
    uniq_blocks = "data/uniq_fixed_blocks.csv"
    df = pd.read_csv(uniq_blocks)
    df.block_rs = df.block_rs.apply(eval)
    print(f"Number of blocks {len(df)}")

    # Build Duplicates with all
    columns = ['block_name', 'block_smi', 'block_r']
    blocks = []
    for irow, (block_smi, block_rs) in df.iterrows():
        added = set()
        for ix, stem in enumerate(block_rs):
            if stem not in added:
                order_block_rs = list(block_rs)
                order_block_rs.pop(ix)
                order_block_rs = [stem] + order_block_rs
                blocks.append([f"{block_smi}_{len(added)}", block_smi, order_block_rs])
                added.update([stem])

    fix_105 = pd.DataFrame(blocks, columns=columns)
    fix_105.to_json(osp.join(datasets_dir, f"fragdb/fix_131.json"))


def build_debug():
    """ Write a debug fragdb with 8 blocks """
    df = pd.read_json(osp.join(datasets_dir, f"fragdb/fix_131.json"))
    # debugsmis = [  # TOP blocks in top10k pre-docked (THIS WAS TOO BIG :( )
    #     "c1ccc2ccccc2c1",
    #     "O=c1nccc[nH]1",
    #     "CNC=O",
    #     "c1ccc2[nH]ccc2c1",
    #     "O=c1nc2[nH]c3ccccc3nc-2c(=O)[nH]1"
    # ]
    debugsmis = [
        # O=C(NCC1CC=C(c2cccc3ccccc23)CC1)n1ccc(-c2cccc3ccccc23)nc1=O
        # oracle -15.6 qed_score 0.301101 synth_score 4.49039
        # mdp_105 [90, 19, 13, 95, 19] | jbonds [[0, 1, 3, 2], [0, 2, 6, 2], [2, 3, 0, 3], [3, 4, 0, 2]]
        'O=c1nccc[nH]1',
        'c1ccc2ccccc2c1',
        'CNC=O',
        'C1=CCCCC1',
        'c1ccc2ccccc2c1'
    ]
    debug = df[df.block_smi.apply(lambda x: x in debugsmis)]
    debug.to_json(osp.join(datasets_dir, f"fragdb/debug_131.json"))
    print(f"REBUILDING DEBUG Fragdb with {len(debugsmis)} uniq blocks and {len(debug)} blocks")


def custom_mdp1():
    """ Generate some custom small debug fragds """
    rows = [
        ["C=O_0", "C=O", [0]],
        ["CO_0", "CO", [0, 1]],
        ["CO_1'", "CO", [1, 0]],
    ]
    df = pd.DataFrame(rows, columns=['block_name', 'block_smi', 'block_r'])
    df.to_json(osp.join(datasets_dir, f"fragdb/test_v0.json"))

    rows = [
        ["C=O_0", "C=O", [0]],
        ["CO_0", "CO", [0, 0]],
    ]
    df = pd.DataFrame(rows, columns=['block_name', 'block_smi', 'block_r'])
    df.to_json(osp.join(datasets_dir, f"fragdb/test_v1.json"))

    # TODO FAILS
    rows = [
        ["C=O_0", "C=O", [0]],
        ["CO_0", "CO", [1, 0, 0]],
        ["CO_1", "CO", [0, 0, 1]],
    ]
    df = pd.DataFrame(rows, columns=['block_name', 'block_smi', 'block_r'])
    df.to_json(osp.join(datasets_dir, f"fragdb/test_v2.json"))

    rows = [
        ["C=O_0", "C=O", [0]], # 0
        ["CO_0", "CO", [1, 0]], # 1
        ["CO_1", "CO", [0, 1]], # 2
        ["C1CCNC1_0", "C1CCNC1", [0, 1, 2]], # 3
        ["C1CCNC1_1", "C1CCNC1", [1, 0, 2]], # 4
        ["C1CCNC1_2", "C1CCNC1", [2, 1, 0]], # 5
    ]
    df = pd.DataFrame(rows, columns=['block_name', 'block_smi', 'block_r'])
    df.to_json(osp.join(datasets_dir, f"fragdb/test_v3.json"))


def load_mdp(bpath):
    """ LOAD MDP for fragdb """
    mdp = MolMDPExtended(bpath)  # Used for generating representation
    mdp_init = {"repr_type": "block_graph", "include_bonds": False}  # block_graph # atom_graph
    mdp_init = getattr(mdp_init, "__dict__", mdp_init)
    mdp_init["device"] = torch.device("cpu")
    mdp.post_init(**mdp_init)

    mdp.build_translation_table()

    # Custom MDP
    num_blocks = len(mdp.block_smi)
    return mdp, num_blocks


def get_random_mol(mdp, maxblocks, msteps):
    mol = BlockMoleculeDataExtended()
    mol = mdp.add_block_to(mol, np.random.randint(maxblocks), 0)
    for _ in range(msteps):
        num_stem_act = mdp.num_act_stems(mol)
        if num_stem_act == 0:
            break
        mol = mdp.add_block_to(mol, np.random.randint(maxblocks), np.random.randint(num_stem_act))
    return mol


def recursive(mdp, mol: BlockMoleculeDataExtended, step, max_steps, num_blocks, seen_graphs=dict()):
    """ get full all all parents """
    children = []

    if len(mol.blockidxs) > 1:
        new = True

        # Let's make things a bit faster
        molid = hashabledict(mdp.unique_id(mol))
        found = seen_graphs.get(molid)
        molg = mdp.get_nx_graph(mol, true_block=True)

        if found is not None:
            # TODO this could be parallelized (will be very slow as the list grows)
            for x, vals in found:
                if mdp.graphs_are_isomorphic(x, molg):
                    new = False
                    vals[0] += 1
                    vals[2].update([mol.smiles])
                    break
        else:
            seen_graphs[molid] = []

        if new:
            seen_graphs[molid].append([molg, [1, len(mdp.parents(mol)), set([mol.smiles])]])
            children.append((molid, mol))

        if step >= max_steps or not new:
            return children

    # for stem in range(mdp.num_act_stems(mol)):
    for stem in range(len(mol.stems)):
        for block in range(num_blocks):
            new_mol = mdp.add_block_to(mol, block, stem)

            new_children = recursive(mdp, new_mol, step+1, max_steps=max_steps,
                                     num_blocks=num_blocks, seen_graphs=seen_graphs)
            children += new_children

    return children


def get_leaves(mdp, get_max_steps, get_num_blocks):
    mol_graphs = dict()
    mol_leaves = []

    for iblock in range(get_num_blocks):
        mol = BlockMoleculeDataExtended()
        mol = mdp.add_block_to(mol, iblock, 0)
        mol_leaves += recursive(mdp, mol, 1, max_steps=get_max_steps, num_blocks=get_num_blocks,
                                seen_graphs=mol_graphs)
        print(f"Finished {iblock}/{get_num_blocks}")

    r_mol_leaves, r_mol_graphs = [], []
    for molid, mol in mol_leaves[::-1]:
        r_mol_leaves.append(mol)
        r_mol_graphs.append(mol_graphs[molid].pop(-1))

    return r_mol_leaves, r_mol_graphs


def test_correct_parents(mol_graphs, show_smiles_duplicates=True):
    print("RUN TEST Compare same number of paths for leaf == len(parents) of leaf ...")
    success = True

    duplicate_smiles = []
    for ix, (molg, (paths, parents, smiles)) in enumerate(mol_graphs):
        if paths != parents:
            success = False
            # print(ix, len(mol_leaves[ix].blockidxs), mol_leaves[ix].blockidxs,
            #       "|", mol_leaves[ix].jbonds, paths, parents)
        if len(smiles) > 1:
            duplicate_smiles.append(list(smiles))

    if show_smiles_duplicates:
        print(f"Found mol with more that 1 smiles: \n {duplicate_smiles}")

    print(f"RUN TEST Compare SUCCESS: "
          f"\n ============== \n |||| {success} |||| \n ==============\n")

    return success


def run_tests_on_small_fragdbs(show_smiles_duplicates=True):
    tests = [
        (osp.join(datasets_dir, f"fragdb/test_v0.json"), 5),
        (osp.join(datasets_dir, f"fragdb/test_v1.json"), 4),
        (osp.join(datasets_dir, f"fragdb/test_v2.json"), 4),
        (osp.join(datasets_dir, f"fragdb/test_v3.json"), 3),
        (osp.join(datasets_dir, f"fragdb/debug_131.json"), 3),
    ]

    start_test_tp = time.time()

    tests_s = []
    for itest, (bpath, max_steps) in enumerate(tests):
        if not os.path.isfile(bpath):
            custom_mdp1()

        print(f"{itest} Running test for {bpath} with max_steps: {max_steps}")
        mdp, num_blocks = load_mdp(bpath)
        mol_leaves, mol_graphs = get_leaves(mdp, max_steps, num_blocks)

        print(f"{itest} Collected {len(mol_leaves)} leaves")

        success = test_correct_parents(mol_graphs, show_smiles_duplicates=show_smiles_duplicates)

        tests_s.append((itest, success))

    alls = all([x[1] for x in tests_s])
    print(f"Total time {time.time() - start_test_tp}")
    print(f"Test succeeded {tests_s}\n")

    if alls:
        print("All tests succeeded!")
    else:
        print("FAILED TEST!!!")


def collect_debug_env(max_steps, load_checkpoint=True, include_all_test=0):
    test = "debug_131"
    bpath = osp.join(datasets_dir, f"fragdb/{test}.json")
    build_debug()

    mdp, get_num_blocks = load_mdp(bpath)

    out_path = f"{datasets_dir}/full_mdp_blocks_{test}_max_steps{max_steps}.pk"
    pout_path = f"{datasets_dir}/_full_mdp_blocks_{test}_max_steps{max_steps}.pk"

    if os.path.isfile(pout_path) and load_checkpoint:
        print("Found checkpoint will load ...")
        ckpt = torch.load(pout_path)
        mol_graphs = ckpt["mol_graphs"]
        mol_leaves = ckpt["mol_leaves"]
        sti = ckpt["partial"] + 1
    else:
        sti = 0
        # Get leaves
        mol_graphs = dict()
        mol_leaves = []

    prev_len = 0
    for iblock in range(sti, get_num_blocks):
        st = time.time()
        print(f'Start {iblock}/{get_num_blocks}')
        sys.stdout.flush()

        mol = BlockMoleculeDataExtended()
        mol = mdp.add_block_to(mol, iblock, 0)
        mol_leaves += recursive(mdp, mol, 1, max_steps=max_steps, num_blocks=get_num_blocks,
                                seen_graphs=mol_graphs)

        t_spent = time.time() - st
        print(f"Finished {iblock}/{get_num_blocks} in {t_spent:.2f}s "
              f"with {len(mol_leaves)} ({len(mol_leaves) - prev_len} new)")
        prev_len = len(mol_leaves)

        torch.save(
            {"mol_graphs": mol_graphs, "mol_leaves": mol_leaves, "partial": iblock},
            pout_path
        )

        print(f"Save partial @ {pout_path}")
        sys.stdout.flush()

    if include_all_test > 0:
        # Test Random molecules that they are included in our leaves list

        different_smiles = []
        for itest in range(include_all_test):
            mol = get_random_mol(mdp, get_num_blocks, np.random.randint(1, max_steps))

            new = True

            # Let's make things a bit faster
            molid = hashabledict(mdp.unique_id(mol))
            found = mol_graphs.get(molid)

            if found is not None:
                molg = mdp.get_nx_graph(mol, true_block=True)

                # TODO this could be parallelized (will be very slow as the list grows)
                for x, vals in found:
                    if mdp.graphs_are_isomorphic(x, molg):
                        new = False
                        if mol.smiles not in vals[-1]:
                            different_smiles.append((mol.smiles))
                        break

            assert not new, f"Not good - found mol not in leaves set" \
                            f"{mol.blockidxs} | {mol.jbonds}"
            if (itest + 1) % (include_all_test // 10) == 0:
                print(f"[Random mol test] {itest} / {include_all_test}")

        print("[Random mol test] Different smiles but considered the same leaf:")
        print(different_smiles)
        print("[Random mol test] Seem to find all samples in our leaf set")

    r_mol_leaves, r_mol_graphs = [], []
    for molid, mol in mol_leaves[::-1]:
        r_mol_leaves.append(mol)
        r_mol_graphs.append(mol_graphs[molid].pop(-1))
    mol_leaves, mol_graphs = r_mol_leaves, r_mol_graphs

    # TEST PARENTS
    success = test_correct_parents(mol_graphs, show_smiles_duplicates=True)
    print(f"Found {len(mol_graphs)} leaves (>1 block)")

    torch.save(
        {"mol_graphs": mol_graphs, "mol_leaves": mol_leaves},
        out_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test some envs if correct parent '
                                                 '/ Save debug env molecules.')
    parser.add_argument('--debug_env', action='store_true', help='Save debug env')
    parser.add_argument('--max_steps', default=4, type=int, help='Max steps for debug env')
    parser.add_argument('--redo', action='store_true', help='Redo debug env')
    parser.add_argument('--test_all', default=100000, type=int, help='Test')
    args = parser.parse_args()

    if not args.debug_env:
        run_tests_on_small_fragdbs(show_smiles_duplicates=True)
    else:
        collect_debug_env(
            args.max_steps, load_checkpoint=(not args.redo), include_all_test=args.test_all
        )
