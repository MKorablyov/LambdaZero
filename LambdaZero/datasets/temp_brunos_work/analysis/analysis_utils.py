from typing import List

import pandas as pd
from chemprop.features import MolGraph
from rdkit.Chem.rdchem import Mol, Atom


def get_all_formal_charges(mol: Mol) -> List[int]:
    return [atom.GetFormalCharge() for atom in mol.GetAtoms()]


def are_two_mols_the_same_for_chemprop(mol1: Mol, mol2: Mol) -> bool:

    g1 = MolGraph(mol1)
    g2 = MolGraph(mol2)

    list_attributes = ['a2b', 'b2a', 'b2revb', 'f_atoms', 'f_bonds', 'n_atoms', 'n_bonds']

    for attribute in list_attributes:
        p1 = getattr(g1, attribute)
        p2 = getattr(g2, attribute)
        if p1 != p2:
            return False
    return True


def get_number_of_hydrogens(mol: Mol) -> List[int]:
    return [atom.GetTotalNumHs() for atom in mol.GetAtoms()]


def get_atom_series(atom: Atom):

    series = pd.Series({'atomic number': atom.GetAtomicNum(),
                        'total degree': atom.GetTotalDegree(),
                        'formal charge': atom.GetFormalCharge(),
                        'chiral tag': atom.GetChiralTag(),
                        'total number of hydrogens': atom.GetTotalNumHs(),
                        'hybridation': atom.GetHybridization(),
                        'is aromatic': atom.GetIsAromatic(),
                        'mass': atom.GetMass()
                        })

    return series