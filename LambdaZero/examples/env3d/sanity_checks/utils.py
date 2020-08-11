import numpy as np

from LambdaZero.examples.env3d.geometry import get_center_of_mass, \
    get_molecular_orientation_vector_from_positions_and_masses
from LambdaZero.examples.env3d.rdkit_utilities import get_atomic_masses


def extract_mol_geometry(anchor_indices, mol, parent_size):
    all_positions = mol.GetConformer().GetPositions()
    all_masses = get_atomic_masses(mol)
    parent_anchor_index, child_anchor_index = anchor_indices
    parent_anchor = all_positions[parent_anchor_index]
    child_anchor = all_positions[child_anchor_index]
    n_axis = get_n_axis(child_anchor, parent_anchor)
    parent_positions, child_positions = np.split(all_positions, [parent_size])
    parent_masses, child_masses = np.split(all_masses, [parent_size])
    parent_cm = get_center_of_mass(parent_masses, parent_positions)
    parent_vector = get_molecular_orientation_vector_from_positions_and_masses(
        parent_masses, parent_positions, parent_anchor, n_axis
    )
    child_cm = get_center_of_mass(child_masses, child_positions)
    child_vector = get_molecular_orientation_vector_from_positions_and_masses(
        child_masses, child_positions, child_anchor, n_axis
    )

    geometry_dict = {
        "all_positions": all_positions,
        "child_anchor": child_anchor,
        "child_cm": child_cm,
        "child_vector": child_vector,
        "n_axis": n_axis,
        "parent_anchor": parent_anchor,
        "parent_cm": parent_cm,
        "parent_vector": parent_vector,
    }

    return geometry_dict


def get_n_axis(child_anchor, parent_anchor):
    n_axis = child_anchor - parent_anchor
    n_axis /= np.linalg.norm(n_axis)
    return n_axis