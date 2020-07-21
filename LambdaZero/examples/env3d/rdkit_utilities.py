from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
from rdkit.Geometry.rdGeometry import Point3D

import numpy as np


def get_atomic_masses(mol: Mol):
    return np.array([a.GetMass() for a in mol.GetAtoms()])


def optimize_mol_in_place(mol: Mol):
    symbol_set = set([a.GetSymbol().lower() for a in mol.GetAtoms()])

    assert "h" not in symbol_set, "can't optimize molecule with h"

    Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94')
    Chem.RemoveHs(mol)


def set_positions_on_conformer(mol: Mol, positions: np.array):
    for i, position in enumerate(positions):
        pt = Point3D()
        pt.x, pt.y, pt.z = position
        mol.GetConformer().SetAtomPosition(i, pt)


def get_mmff_force_field(mol: Mol, confId: int=-1):
    """
    This hack is necessary to pass the correct information to the force field constructor
    """
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94')

    return AllChem.MMFFGetMoleculeForceField(mol, properties, confId)


def get_mmff_energy(mol_with_hydrogens: Mol, conf_id: int = 0):
    properties = AllChem.MMFFGetMoleculeProperties(mol_with_hydrogens, mmffVariant='MMFF94')
    energy = AllChem.MMFFGetMoleculeForceField(mol_with_hydrogens, properties, confId=conf_id).CalcEnergy()
    return energy


def find_index_of_lowest_converged_energy(list_tuples_energy_converged: List[Tuple]) -> int:
    """
    Finds the index of the lowest converged energy

    Args:
        list_tuples_energy_converged (list(tuple)): the output of  MMFFOptimizeMoleculeConfs

    Returns:
        lowest_converged_energy_index (int): index of lowest converged energy

    """
    mask = np.array([t[0] != 0 for t in list_tuples_energy_converged])  # a flag value of 0 means "converged"

    if np.all(mask):  # nothing is converged
        return np.NaN

    energies = np.array([t[1] for t in list_tuples_energy_converged])

    masked_energies = np.ma.masked_array(energies, mask=mask)
    lowest_energy_index = np.argmin(masked_energies)

    return lowest_energy_index

