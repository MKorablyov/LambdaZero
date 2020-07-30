import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem.rdchem import Atom
from rdkit.Chem.rdmolfiles import MolFromSmiles

from LambdaZero.environments import MolMDP
from LambdaZero.examples.env3d.dataset.data_generation import (
    get_smiles_and_consistent_positions,
    get_blocks_embedding_energies,
)
from LambdaZero.examples.env3d.rdkit_utilities import (
    set_positions_on_conformer,
    get_atomic_symbols,
)


@pytest.fixture
def mol_positions_attachment_index(blocks_file):
    random_seed = 333
    np.random.seed(random_seed)
    number_of_blocks = 5

    mdp = MolMDP(blocks_file=blocks_file)
    mdp.reset()
    mdp.random_walk(number_of_blocks)

    number_of_atoms = mdp.molecule.mol.GetNumAtoms()

    positions = np.random.rand(number_of_atoms, 3)

    attachment_index = np.random.choice(number_of_atoms)

    # use a unique atom which is not present in the blocks to uniquely tag the attachment atom
    silicon = Atom("Si")
    assert "Si" not in set(
        get_atomic_symbols(mdp.molecule.mol)
    ), "Silicon is present in the molecule! Review."

    editable_mol = Chem.EditableMol(mdp.molecule.mol)
    editable_mol.ReplaceAtom(attachment_index, silicon)
    mol = editable_mol.GetMol()

    return mol, positions, attachment_index


def test_get_smiles_and_consistent_positions(mol_positions_attachment_index):
    mol, positions, attachment_index = mol_positions_attachment_index

    smiles, permuted_positions, permuted_attachment_index = get_smiles_and_consistent_positions(
        mol, positions, attachment_index
    )

    EmbedMolecule(mol)
    set_positions_on_conformer(mol, positions)

    smiles_mol = MolFromSmiles(smiles)
    EmbedMolecule(smiles_mol)
    set_positions_on_conformer(smiles_mol, permuted_positions)

    # test that the root mean square distance is vanishingly small when we align the two molecules
    rmsd = AlignMol(mol, smiles_mol)

    np.testing.assert_almost_equal(rmsd, 0.0)

    # Test that the attachment atom is the same before and after the permutation. We assume that
    # the data has been prepared so that this atom type is unique in the molecule.
    mol_attached_atom = mol.GetAtomWithIdx(attachment_index).GetSymbol()
    smiles_mol_attached_atom = smiles_mol.GetAtomWithIdx(
        permuted_attachment_index
    ).GetSymbol()

    assert mol_attached_atom == smiles_mol_attached_atom


def test_get_blocks_embedding_energies(blocks_file):
    blocks_df = pd.read_json(blocks_file)
    computed_energy_dictionary = get_blocks_embedding_energies(blocks_file)

    for block_index, row in blocks_df.iterrows():

        # check that the energy for one atom blocks is zero.
        if MolFromSmiles(row["block_smi"]).GetNumAtoms() == 1:
            assert computed_energy_dictionary[block_index] == 0.0

    for smiles, group_df in blocks_df.groupby(by="block_smi"):
        if len(group_df) > 1:
            same_smiles_indices = list(group_df.index)
            same_smiles_energies = [
                computed_energy_dictionary[block_index]
                for block_index in same_smiles_indices
            ]

            # check that all energies for the same smiles are the same.
            assert len(np.unique(same_smiles_energies)) == 1.0
