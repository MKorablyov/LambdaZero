from chemprop.features import MolGraph
from rdkit.Chem.rdmolfiles import MolFromSmiles

from LambdaZero.chem import mol_to_graph
from LambdaZero.datasets.brutal_dock.models.chemprop_adaptor import graph_to_mol


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
