from typing import List, Tuple, Set

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol, RWMol
from rdkit.Chem.rdchem import BondType
from torch_geometric.data import Data
import numpy as np


# By inspection of file LambdaZero/chem/chem_op.py, the atomic number is encoded
# as a value in the node features of the graph with this index
_ATOMIC_NUMBER_INDEX = 6

# By inspection of file LambdaZero/chem/chem_op.py, the number of hydrogens on each atom is encoded
# as a value in the node features of the graph with this index
_NUMBER_OF_HYDROGEN_INDEX = 13

# By inspection of file LambdaZero/chem/chem_op.py, the relationship
# between the RDKit bond type and the onehot encoded data is given by this.
_BONDTYPE_MAPPING = {0: BondType.SINGLE, 1: BondType.DOUBLE, 2: BondType.TRIPLE, 3: BondType.AROMATIC}


def _get_integers_from_node_features(molecule_graph: Data, index: int) -> List[int]:
    # RDKit is picky about types. This must really be ints, not numpy.int64
    integer_features = [int(a) for a in molecule_graph.x[:, index].numpy().astype(int)]
    return integer_features


def _get_atomic_numbers_from_molecular_graph(molecule_graph: Data) -> List[int]:
    return _get_integers_from_node_features(molecule_graph, _ATOMIC_NUMBER_INDEX)


def _get_number_of_hydrogens_from_molecular_graph(molecule_graph: Data) -> List[int]:
    return _get_integers_from_node_features(molecule_graph, _NUMBER_OF_HYDROGEN_INDEX)


def _assert_that_each_atom_pair_appears_only_once(set_of_bonds: Set[Tuple[int, int, BondType]]):
    pairs = [(idx1, idx2) for idx1, idx2, _ in set_of_bonds]
    assert len(set(pairs)) == len(pairs), "a pair of indices appears more than once!"



def _get_bonds_from_molecular_graph(molecule_graph: Data) -> Set[Tuple[int, int, BondType]]:
    bond_atom_index_pairs = molecule_graph.edge_index.t().numpy().astype(int)
    bond_atom_index_pairs.sort(axis=1)
    one_hot_encoded_bond_types = molecule_graph.edge_attr.numpy().astype(int)

    _, bond_type_indices = np.where(one_hot_encoded_bond_types == 1)
    bond_types = [_BONDTYPE_MAPPING[bond_type_index] for bond_type_index in bond_type_indices]

    set_of_bonds = set()
    for (idx1, idx2), bond_type in zip(bond_atom_index_pairs, bond_types):

        set_of_bonds.add((idx1, idx2, bond_type))

    _assert_that_each_atom_pair_appears_only_once(set_of_bonds)

    return set_of_bonds


def graph_to_mol(molecule_graph: Data) -> Mol:
    """
    This method creates an RDKit Mol object from a pytorch_geometric molecular graph.
    The method used is based on https://sourceforge.net/p/rdkit/mailman/message/34135849/
    """
    atomic_numbers = _get_atomic_numbers_from_molecular_graph(molecule_graph)

    # TODO: I'm trying to force the hydrogen count to be correct, but it seems to have no effect
    hydrogen_counts = _get_number_of_hydrogens_from_molecular_graph(molecule_graph)

    bonds = _get_bonds_from_molecular_graph(molecule_graph)

    blank_molecule = Mol()  # creates a blank molecule, can use an existing RDKit
    editable_molecule = RWMol(blank_molecule)

    list_molecule_indices = []
    for atomic_number, number_of_hydrogens in zip(atomic_numbers, hydrogen_counts):
        atom = Chem.Atom(atomic_number)

        # TODO: This seems to be ineffective
        atom.SetNumExplicitHs(number_of_hydrogens)

        idx = editable_molecule.AddAtom(atom)
        list_molecule_indices.append(idx)

    for atom_index_1, atom_index_2, bond_type in bonds:
        idx1 = list_molecule_indices[atom_index_1]
        idx2 = list_molecule_indices[atom_index_2]
        _ = editable_molecule.AddBond(idx1, idx2, bond_type)

    molecule = editable_molecule.GetMol()
    Chem.SanitizeMol(molecule)
    AllChem.EmbedMolecule(molecule)
    AllChem.MMFFOptimizeMolecule(molecule)

    new_molecule = RWMol(molecule)

    for atom, number_of_hydrogens in zip(new_molecule.GetAtoms(), hydrogen_counts):
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(number_of_hydrogens)

    molecule = editable_molecule.GetMol()
    [atom.SetNoImplicit(True) for atom in molecule.GetAtoms()]

    Chem.SanitizeMol(molecule)
    AllChem.EmbedMolecule(molecule)
    AllChem.MMFFOptimizeMolecule(molecule)

    return molecule
