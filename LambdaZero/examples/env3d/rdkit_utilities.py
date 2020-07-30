from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolops import SanitizeMol
from rdkit.Geometry.rdGeometry import Point3D

import numpy as np


def get_atomic_symbols(mol: Mol):
    return [a.GetSymbol() for a in mol.GetAtoms()]


def get_atomic_masses(mol: Mol):
    return np.array([a.GetMass() for a in mol.GetAtoms()])


def optimize_mol_in_place(mol: Mol):
    symbol_set = set([a.GetSymbol().lower() for a in mol.GetAtoms()])

    assert "h" not in symbol_set, "can't optimize molecule with h"

    Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94")
    Chem.RemoveHs(mol)


def set_positions_on_conformer(mol: Mol, positions: np.array):
    for i, position in enumerate(positions):
        pt = Point3D()
        pt.x, pt.y, pt.z = position
        mol.GetConformer().SetAtomPosition(i, pt)


def get_mmff_force_field(mol: Mol, confId: int = -1):
    """
    This hack is necessary to pass the correct information to the force field constructor
    """
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94")

    return AllChem.MMFFGetMoleculeForceField(mol, properties, confId)


def get_mmff_energy(mol_with_hydrogens: Mol, conf_id: int = 0):
    properties = AllChem.MMFFGetMoleculeProperties(
        mol_with_hydrogens, mmffVariant="MMFF94"
    )
    energy = AllChem.MMFFGetMoleculeForceField(
        mol_with_hydrogens, properties, confId=conf_id
    ).CalcEnergy()
    return energy


def find_index_of_lowest_converged_energy(
    list_tuples_energy_converged: List[Tuple]
) -> int:
    """
    Finds the index of the lowest converged energy

    Args:
        list_tuples_energy_converged (list(tuple)): the output of  MMFFOptimizeMoleculeConfs

    Returns:
        lowest_converged_energy_index (int): index of lowest converged energy

    """
    mask = np.array(
        [t[0] != 0 for t in list_tuples_energy_converged]
    )  # a flag value of 0 means "converged"

    if np.all(mask):  # nothing is converged
        return np.NaN

    energies = np.array([t[1] for t in list_tuples_energy_converged])

    masked_energies = np.ma.masked_array(energies, mask=mask)
    lowest_energy_index = int(
        np.argmin(masked_energies)
    )  # cast to int because RDKIT is PICKY.

    return lowest_energy_index


def get_lowest_energy_and_mol_with_hydrogen(
    mol: Mol, num_conf: int, max_iters: int = 500, random_seed: int = 0
):
    """
    Convenience method to add hydrogens, embed num_conf versions, optimize positions and extract lowest
    energy conformer.

    Args:
        mol (Mol): a Mol object
        num_conf (int): number of conformers to embed
        max_iters (int): maximum number of iterations for the optimizer to converge
        random_seed (int): random seed for the embedding

    Returns:
        min_energy (float): lowest converged energy
        mol_with_hydrogen (Mol):  mol object, with hydrogens and a single conformer corresponding to
                                  the lowest energy.
        number_of_successes (int): how many optimizations converged

    """
    mol_with_hydrogen = Chem.AddHs(mol)
    SanitizeMol(mol_with_hydrogen)
    AllChem.EmbedMultipleConfs(
        mol_with_hydrogen, numConfs=num_conf, randomSeed=random_seed
    )
    list_tuples_energy_converged = AllChem.MMFFOptimizeMoleculeConfs(
        mol_with_hydrogen, mmffVariant="MMFF94", maxIters=max_iters
    )
    lowest_energy_index = find_index_of_lowest_converged_energy(
        list_tuples_energy_converged
    )

    number_of_successes = np.sum([t[0] == 0 for t in list_tuples_energy_converged])
    if number_of_successes == 0:
        raise ValueError("No conformation converged. Review input parameters")

    min_energy = list_tuples_energy_converged[lowest_energy_index][1]

    conformer_indices_to_remove = list(range(num_conf))
    conformer_indices_to_remove.pop(lowest_energy_index)
    _ = [mol_with_hydrogen.RemoveConformer(i) for i in conformer_indices_to_remove]

    return min_energy, mol_with_hydrogen, number_of_successes


def write_conformer_to_file(filepath, mol, conf_id=0):
    with open(filepath, "w") as f:
        print(Chem.MolToMolBlock(mol, confId=conf_id), file=f)


def get_index_permutation_between_equivalent_molecules(
    original_mol: Mol, smiles_mol: Mol
):
    """

    The atoms in a Mol object are identified by indices. It is possible for the exact same molecule
    to be represented by two nominally different Mol objects that only differ in how the atoms are indexed.
    In particular, a Mol obtained from molMDP.random_walk and the Mol obtained from the corresponding smiles
    string are not ordered the same way in general.

    This method first validates that the two Mols are indeed the same up to index permutation, and computes
    the index permutation that takes the original_mol indexing to the smiles_mol indexing.

    Args:
        original_mol (Mol): molecule represented as a Mol
        smiles_mol (Mol): same molecule, but derived from the corresponding smiles string

    Returns:
        permutation (np.array): Array of indices such that, for p_i = permutation[i],
                                the atom with index "i" in smiles_mol is the same as atom with index "p_i" in mol.

    Example:

        consider atoms ordered in the following way
            orignal_mol = [Carbon, Nitrogen, Sulfur]
            smiles_mol = [Sulfur, Carbon, Nitrogen]

        Then permutation = [2, 0, 1], which is to say:
            - the Sulfur in position 0 in smiles_mol comes from position 2 in original_mol
            - the Carbon in position 1 in smiles_mol comes from position 0 in original_mol
            - the Nitrogen in position 2 in smiles_mol comes from position 1 in original_mol

    """

    assert (
        smiles_mol.GetNumAtoms() == original_mol.GetNumAtoms()
    ), "the two molecules do not have the same number of atoms"

    # The documentation for Mol.GetSubstructMatch states:
    #    the ordering of the indices corresponds to the atom ordering
    #    in the query. For example, the first index is for the atom in
    #    this molecule that matches the first atom in the query.

    match = original_mol.GetSubstructMatch(smiles_mol)

    expected_index_set = set(range(smiles_mol.GetNumAtoms()))
    computed_index_set = set(match)

    assert (
        expected_index_set == computed_index_set
    ), "There is no perfect match between original_mol and smiles_mol: review input!"

    permutation = np.array(match)
    return permutation
