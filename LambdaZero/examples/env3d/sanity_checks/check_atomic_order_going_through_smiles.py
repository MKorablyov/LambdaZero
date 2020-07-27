"""
The goal of this script is to check that atomic order are preserved by going to the SMILES
representation.

This script shows that ATOMIC ORDER IS NOT PRESERVED BY GOING TO THE SMILES REPRESENTATION;
The assert at the end of the script FAILS.

"""
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles

import LambdaZero.utils
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.molecular_connection import MolecularConnection

datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

total_number_of_blocks = 5
number_of_iterations = 100

if __name__ == "__main__":

    np.random.seed(0)
    molMDP = MolMDP(blocks_file=blocks_file)
    molMDP.reset()
    molMDP.random_walk(total_number_of_blocks)

    original_molecule = molMDP.molecule.mol

    smiles = Chem.MolToSmiles(original_molecule)
    molecule_from_smiles = MolFromSmiles(smiles)

    original_connection = MolecularConnection(original_molecule)
    smiles_connection = MolecularConnection(molecule_from_smiles)

    assert original_connection.child_is_consistent(
        smiles_connection
    ), "Molecule is NOT CONSISTENT with the equivalent molecule generated from its smiles"
