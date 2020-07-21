"""
The goal of this script is to verify that atoms are in the same order as multiple blocks are added to
the molecule.
"""
import os

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.rdkit_utilities import optimize_mol_in_place


class MolecularConnection:
    def __init__(self, mol: Mol):

        atoms = mol.GetAtoms()

        self.nodes = []
        self.atomic_type = []
        self.neighbors = []

        for a in atoms:
            self.nodes.append(a.GetIdx())
            self.atomic_type.append(a.GetAtomicNum())
            self.neighbors.append(set(n.GetIdx() for n in a.GetNeighbors()))

        self.size = len(self.nodes)

        self.node_set = set(self.nodes)

    def _have_same_nodes(self, child):
        return self.nodes == child.nodes[: self.size]

    def _have_same_atomic_types(self, child):
        return self.atomic_type == child.atomic_type[: self.size]

    def _neighborhood_sets_are_consistent(
        self, parent_neighbors_set, child_neighors_set
    ):
        difference_set = child_neighors_set.difference(parent_neighbors_set)
        return difference_set.isdisjoint(self.node_set)

    def child_is_consistent(self, child):
        if not self._have_same_nodes(child) or not self._have_same_atomic_types(child):
            return False

        for parent_neighbors_set, child_neighors_set in zip(
            self.neighbors, child.neighbors[: self.size]
        ):
            if not self._neighborhood_sets_are_consistent(
                parent_neighbors_set, child_neighors_set
            ):
                return False

        return True


def get_intermediate_mol(number_of_blocks: int):
    mol, _ = mol_from_frag(
        jun_bonds=molMDP.molecule.jbonds[: number_of_blocks - 1],
        frags=molMDP.molecule.blocks[:number_of_blocks],
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

    for _ in range(number_of_iterations):
        molMDP.reset()
        molMDP.random_walk(total_number_of_blocks)
        smiles = Chem.MolToSmiles(molMDP.molecule.mol)

        print(f"Checking all descendants in {smiles} are consistent...")

        mol = get_intermediate_mol(number_of_blocks=1)
        parent = MolecularConnection(mol)

        for number_of_blocks in range(2, total_number_of_blocks):
            mol = get_intermediate_mol(number_of_blocks)
            child = MolecularConnection(mol)

            are_consistent = parent.child_is_consistent(child)

            assert (
                are_consistent
            ), "Parent and Child are not consistent! Atomic orders get scrambled!"

    print("All parents and children sampled are consistent!")
