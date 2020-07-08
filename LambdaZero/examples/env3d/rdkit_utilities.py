from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import numpy as np


def get_atomic_masses(mol: Mol):
    return np.array([a.GetMass() for a in mol.GetAtoms()])


def optimize_mol_in_place(mol: Mol):
    symbol_set = set([a.GetSymbol().lower() for a in mol.GetAtoms()])

    assert "h" not in symbol_set, "can't optimize molecule with h"

    Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.RemoveHs(mol)
