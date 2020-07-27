"""
The goal of this script is to verify that atoms are in the same order as multiple blocks are added to
the molecule.
"""
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag
from LambdaZero.environments import molMDP
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.molecular_connection import MolecularConnection
from LambdaZero.examples.env3d.rdkit_utilities import optimize_mol_in_place


def get_intermediate_mol(mol_mdp: molMDP, number_of_blocks: int):
    mol, _ = mol_from_frag(
        jun_bonds=mol_mdp.molecule.jbonds[: number_of_blocks - 1],
        frags=mol_mdp.molecule.blocks[:number_of_blocks],
        optimize=False,
    )
    optimize_mol_in_place(mol)
    return mol


datasets_dir, _, _ = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

total_number_of_blocks = 5
number_of_iterations = 100

if __name__ == "__main__":

    np.random.seed(0)

    molMDP = MolMDP(blocks_file=blocks_file)

    for _ in tqdm(range(number_of_iterations)):
        molMDP.reset()
        molMDP.random_walk(total_number_of_blocks)

        original_molecule = molMDP.molecule.mol
        smiles = Chem.MolToSmiles(original_molecule, rootedAtAtom=0)
        molecule_from_smiles = MolFromSmiles(smiles)

        original_connection = MolecularConnection(original_molecule)
        smiles_connection = MolecularConnection(molecule_from_smiles)

        #print(f"Checking all descendants in {smiles} are consistent...")

        mol = get_intermediate_mol(molMDP, number_of_blocks=1)
        parent = MolecularConnection(mol)

        for number_of_blocks in range(2, total_number_of_blocks):
            mol = get_intermediate_mol(molMDP, number_of_blocks)
            child = MolecularConnection(mol)

            are_consistent = parent.child_is_consistent(child)

            assert (
                are_consistent
            ), "Parent and Child are not consistent! Atomic orders get scrambled!"

    print("All parents and children sampled are consistent!")
