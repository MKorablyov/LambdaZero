"""
The goal of this script is to reconstruct the child molecule from the parent + child block, confirming
that we obtain the correct answer.
"""
import os

import numpy as np
from rdkit.Chem.rdchem import Mol
import pandas as pd
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.molecular_connection import MolecularConnection
from LambdaZero.examples.env3d.sanity_checks.utils import get_child_molecule

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

number_of_blocks = 5
random_seed = 2342

number_of_samples = 1000

if __name__ == "__main__":
    np.random.seed(random_seed)

    blocks_df = pd.read_json(blocks_file)

    molMDP = MolMDP(blocks_file=blocks_file)
    for _ in tqdm(range(number_of_samples)):

        molMDP.reset()
        molMDP.random_walk(number_of_blocks)

        # extract the child molecule directly from the MDP
        expected_child_mol = Mol(molMDP.molecule.mol)
        expected_connection = MolecularConnection(expected_child_mol)

        # build the child molecule with mol_from_frag; check it is the same as
        # the directly obtained molecule
        test_child_mol, bond = mol_from_frag(
            jun_bonds=molMDP.molecule.jbonds,
            frags=molMDP.molecule.blocks,
            optimize=False,
        )
        test_connection = MolecularConnection(test_child_mol)
        assert test_connection == expected_connection, \
            "the output of the random walk is not consistent with mol_from_frag"

        # extract the parent molecule, which is composed of the N-1 first blocks,
        # and the attchment node index.
        parent_anchor_index = bond[-1][0]
        parent_mol, _ = mol_from_frag(
            jun_bonds=molMDP.molecule.jbonds[: number_of_blocks-2],
            frags=molMDP.molecule.blocks[:number_of_blocks-1],
            optimize=False,
        )
        child_block_index = molMDP.molecule.blockidxs[-1]

        computed_child_mol = get_child_molecule(child_block_index, parent_anchor_index, blocks_df)

        computed_connection = MolecularConnection(computed_child_mol)

        assert expected_connection == computed_connection, \
            "the reconstructed child molecule is inconsistent"

    print(f"the reconstruction is consistent for {number_of_samples} samples")
