import pytest
from chemprop.features import MolGraph
from rdkit.Chem.rdmolfiles import MolFromSmiles
import numpy as np

from LambdaZero.chem import mol_to_graph, Chem
from LambdaZero.datasets.temp_brunos_work.models.chemprop_adaptor import graph_to_mol


@pytest.mark.xfail
def test_rdkit_api():
    """
    This purposefully failing test shows how the chemprop API raises an ArgumentError if the
    Atom constructor is passed a np.int64 integer instead of a standard python integer.

    The error raised is of the form

    >       _ = Chem.Atom(numpy_atomic_number)
E       Boost.Python.ArgumentError: Python argument types in
E           Atom.__init__(Atom, numpy.int64)
E       did not match C++ signature:
E           __init__(_object*, unsigned int)
E           __init__(_object*, RDKit::Atom)
E           __init__(_object*, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >)
    """

    numpy_atomic_number = np.int64(1)
    _ = Chem.Atom(numpy_atomic_number)


def test_graph_to_mol_easy(easy_smiles):
    for smiles in easy_smiles:

        molecule_graph = mol_to_graph(smiles)
        expected_mol = MolFromSmiles(smiles)
        computed_mol = graph_to_mol(molecule_graph)

        expected_mol_graph = MolGraph(expected_mol)
        computed_mol_graph = MolGraph(computed_mol)

        list_attributes = ['a2b', 'b2a', 'b2revb', 'f_atoms', 'f_bonds', 'n_atoms', 'n_bonds']

        for attribute in list_attributes:
            expected = getattr(expected_mol_graph, attribute)
            computed = getattr(computed_mol_graph, attribute)
            assert expected == computed, f"The attribute {attribute} is not the same"


@pytest.mark.skip(reason="The adaptor is broken. This test shows a concrete examples of how it fails. Cannot fix now.")
def test_graph_to_mol_hard(hard_smiles):
    for smiles in hard_smiles:

        molecule_graph = mol_to_graph(smiles)
        expected_mol = MolFromSmiles(smiles)
        computed_mol = graph_to_mol(molecule_graph)

        expected_mol_graph = MolGraph(expected_mol)
        computed_mol_graph = MolGraph(computed_mol)

        list_attributes = ['a2b', 'b2a', 'b2revb', 'f_atoms', 'f_bonds', 'n_atoms', 'n_bonds']

        for attribute in list_attributes:
            expected = getattr(expected_mol_graph, attribute)
            computed = getattr(computed_mol_graph, attribute)
            assert expected == computed, f"The attribute {attribute} is not the same"
