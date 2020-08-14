from rdkit.Chem.rdchem import Mol


class MolecularConnection(object):
    """
    This class builds a data structure to compare two Mol objects with respect to the
    order of their constituting atoms.

    for
       parent_connection  = MolecularConnection(parent_mol)
       child_connection  = MolecularConnection(child_mol)

    if
        parent_connection.child_is_consistent(child_connection) = False
    then the parent and child molecule are not consistent.
    if
        parent_connection.child_is_consistent(child_connection) = True
    it does not constitute a proof that they are consistent, but it's a big hint that they are.

    """
    def __eq__(self, other):
        same_nodes = self.nodes == other.nodes
        same_types = self.atomic_type == other.atomic_type
        same_neighbors = self.neighbors == other.neighbors

        return same_nodes and same_types and same_neighbors

    def __init__(self, mol: Mol):
        """
        Instantiates various data structures that will be used to compare two molecules for consistency.

        Args:
            mol (Mol): a Mol object
        """

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
        """
        Check if the parent and child have the same nodes, up to the dimension of the parent.

        Args:
            child (MolecularConnection): a molecular connection object for another molecule

        Returns:
            have_the_same_nodes (Bool): whether the node indices are the same up to self.size

        """
        return self.nodes == child.nodes[: self.size]

    def _have_same_atomic_types(self, child):
        """
        Check if the parent and child have the same atomic types, up to the dimension of the parent.

        Args:
            child (MolecularConnection): a molecular connection object for another molecule

        Returns:
            have_the_same_atomic_types (Bool): whether the node atoms are the same up to self.size

        """
        return self.atomic_type == child.atomic_type[: self.size]

    def _neighborhood_sets_are_consistent(
        self, parent_neighbors_set, child_neighors_set
    ):
        """
        Check if the child neighborood is a superset of the parent neibhboorhood.

        Args:
            parent_neighbors_set (set): set of node indices that are the neighbor of a specific node in the parent
            child_neighors_set (set): set of node indices that are the neighbor of a specific node in the child

        Returns:
            Bool: True if the child neighborhood contains the parent neighborhood. Note that the child
                  neighborhood could be larger than the parent neighborhood because there are extra bonds
                  joining the child to the parent.
        """
        difference_set = child_neighors_set.difference(parent_neighbors_set)
        return difference_set.isdisjoint(self.node_set)

    def child_is_consistent(self, child):
        """
        Check if parent and child molecules are consistent, up to size of parent molecule.

        Args:
            child (MolecularConnection): a molecular connection object for another molecule

        Returns:
            Bool: is the child molecule consistent with the parent molecule

        """
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
