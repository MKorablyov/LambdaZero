from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Mol
import numpy as np
from rdkit.Geometry.rdGeometry import Point3D


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