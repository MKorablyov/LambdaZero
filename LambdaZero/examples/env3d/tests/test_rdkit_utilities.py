import pytest
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles

from LambdaZero.examples.env3d.rdkit_utilities import (
    find_index_of_lowest_converged_energy,
    get_index_permutation_between_equivalent_molecules,
)


@pytest.fixture
def original_mol():
    """
    Create a simple, not necessarily realistic molecule
    """

    editable_mol = Chem.EditableMol(Chem.Mol())

    c_index = editable_mol.AddAtom(Atom("C"))
    n_index = editable_mol.AddAtom(Atom("N"))
    s_index = editable_mol.AddAtom(Atom("S"))

    editable_mol.AddBond(c_index, s_index)
    editable_mol.AddBond(s_index, n_index)

    original_mol = editable_mol.GetMol()

    return original_mol


@pytest.fixture
def inconsistent_mols():
    """
    Create mols that are not consistent with original_mol, to test error catching code.
    """

    editable_mol = Chem.EditableMol(Chem.Mol())

    c_index = editable_mol.AddAtom(Atom("C"))
    p_index = editable_mol.AddAtom(Atom("P"))
    s_index = editable_mol.AddAtom(Atom("S"))

    editable_mol.AddBond(c_index, s_index)
    editable_mol.AddBond(s_index, p_index)

    inconsistent_mol1 = editable_mol.GetMol()

    editable_mol = Chem.EditableMol(Chem.Mol())

    c_index = editable_mol.AddAtom(Atom("C"))
    s_index = editable_mol.AddAtom(Atom("S"))

    editable_mol.AddBond(c_index, s_index)

    inconsistent_mol2 = editable_mol.GetMol()

    return [inconsistent_mol1, inconsistent_mol2]


@pytest.fixture
def smiles_mol(original_mol):
    smiles = MolToSmiles(original_mol)
    smiles_mol = MolFromSmiles(smiles)
    return smiles_mol


@pytest.fixture
def expected_permutation(original_mol, smiles_mol):
    """
    We assume all atoms are different for simplicity.
    """
    smiles_atoms = smiles_mol.GetAtoms()
    original_atoms = original_mol.GetAtoms()

    permutation = []
    for original_atom in original_atoms:
        for smiles_index, smiles_atom in enumerate(smiles_atoms):
            if smiles_atom.GetSymbol() == original_atom.GetSymbol():
                permutation.append(smiles_index)
                continue

    return np.array(permutation)


list_clean_tuples = [(1.0, 1.0), (0.0, 3.0), (1, 3.0), (0.0, 2.0), (1.0, 1.0)]
clean_index = 3

list_bad_tuples = [(1.0, 1.0), (1.0, 2.0)]
bad_index = np.NaN


@pytest.mark.parametrize(
    "energy_converged_tuples, expected_lowest_energy_index",
    [(list_clean_tuples, clean_index), (list_bad_tuples, bad_index)],
)
def test_find_index_of_lowest_converged_energy(
    energy_converged_tuples, expected_lowest_energy_index
):
    computed_lowest_energy_index = find_index_of_lowest_converged_energy(
        energy_converged_tuples
    )
    np.testing.assert_equal(computed_lowest_energy_index, expected_lowest_energy_index)


def test_get_index_permutation_between_equivalent_mol_assert(
    original_mol, inconsistent_mols
):
    for inconsistent_mol in inconsistent_mols:
        with pytest.raises(AssertionError):
            get_index_permutation_between_equivalent_molecules(
                original_mol, inconsistent_mol
            )


def test_get_index_permutation_between_equivalent_mol(
    original_mol, smiles_mol, expected_permutation
):
    computed_permutation = get_index_permutation_between_equivalent_molecules(
        original_mol, smiles_mol
    )
    np.testing.assert_array_equal(computed_permutation, expected_permutation)
