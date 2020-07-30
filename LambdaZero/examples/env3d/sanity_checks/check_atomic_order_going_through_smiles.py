"""
The goal of this script is to check that atomic order are preserved by going to the SMILES
representation.

This script shows that :
    1. Smiles -> Mol generates a consistent atomic order, ie, the same order if we perform the
       same operation multiple times.

    2. THE ATOMIC ORDER OF molMDP.random_walk IS NOT CONSISTENT WITH THE ORDER OF THE SMILES REPRESENTATION
        --> The assert at the end of the script FAILS.
        --> This has implications for how we generate the dataset; the order of atoms must be consistent since the
            molecule is passed as a smiles and the atomic coordinates assume a fixed, known atomic order.
"""
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.molecular_connection import MolecularConnection

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

total_number_of_blocks = 5
number_of_iterations = 500

if __name__ == "__main__":

    np.random.seed(0)
    molMDP = MolMDP(blocks_file=blocks_file)

    # Check that the order of atoms remains fixed when we go SMILES -> Mol
    for _ in tqdm(range(number_of_iterations)):
        molMDP.reset()
        molMDP.random_walk(total_number_of_blocks)

        # perform two cycles of Mol -> smiles,  smiles -> Mol
        molecule_from_random_walk = molMDP.molecule.mol
        smiles1 = Chem.MolToSmiles(molecule_from_random_walk)
        mol1_from_smiles = MolFromSmiles(smiles1)

        smiles2 = Chem.MolToSmiles(mol1_from_smiles)
        mol2_from_smiles = MolFromSmiles(smiles2)

        assert (
            smiles1 == smiles2
        ), "subsequent SMILES for the same molecule are not the same"

        connection1 = MolecularConnection(mol1_from_smiles)
        connection2 = MolecularConnection(mol2_from_smiles)

        assert connection1.child_is_consistent(
            connection2
        ) and connection2.child_is_consistent(
            connection1
        ), "Two molecules from the same SMILES are not order-consistent"

    print("The atom order in Mol is consistent when we go from smiles to Mol.")

    # Check the atomic order for a mol generated from molMDP.randomwalk vs. the order from its smiles
    molMDP.reset()
    molMDP.random_walk(total_number_of_blocks)

    molecule_from_random_walk = molMDP.molecule.mol

    smiles = Chem.MolToSmiles(molecule_from_random_walk)
    molecule_from_smiles = MolFromSmiles(smiles)

    random_walk_connection = MolecularConnection(molecule_from_random_walk)
    smiles_connection = MolecularConnection(molecule_from_smiles)

    # I expect this assert to FAIL. Danger, danger!!
    assert random_walk_connection.child_is_consistent(smiles_connection), (
        "The order of atoms for a Mol generated with molMDP.random_walk is NOT CONSISTENT "
        "with the atomic order of the equivalent molecule generated from its smiles."
    )
