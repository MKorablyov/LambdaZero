import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from LambdaZero.chem import mol_from_frag
from LambdaZero.examples.env3d.geometry import (
    get_center_of_mass,
    get_molecular_orientation_vector_from_positions_and_masses,
    get_n_axis,
)
from LambdaZero.examples.env3d.rdkit_utilities import get_atomic_masses


def extract_mol_geometry(anchor_indices, mol, parent_size):
    """
    Extracts verious geometric vectors of interest such as centers of mass, directions, axis of rotation, etc...

    Args:
        anchor_indices (Tuple[int, int]): indices of parent and child atoms that bind the child block to parent
        mol (Mol): Mol object for child molecule, assumed to have a conformer
        parent_size (int): number of atoms in parent molecule

    Returns:
       geometry_dict (Dict): all relevant information in dict format.

    """
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
        "n_axis": n_axis,
        "child_anchor": child_anchor,
        "child_cm": child_cm,
        "child_vector": child_vector,
        "child_positions": child_positions,
        "parent_anchor": parent_anchor,
        "parent_cm": parent_cm,
        "parent_vector": parent_vector,
        "parent_positions": parent_positions,
    }

    return geometry_dict


def get_child_molecule(parent_mol: Mol, child_block_index: int, parent_anchor_index: int, blocks_df: pd.DataFrame):
    """
    This method reconstructs the child molecule given the parent molecule, the attachment node index
    and the vocabulary from which to extract block information.
    Args:
        parent_mol (Mol): parent molecule
        child_block_index (int): child block identifier
        parent_anchor_index (int):  index of the parent node where the child block is attached
        blocks_df (pd.DataFrame): vocabulary of blocks, in Dataframe form

    Returns:

        child_mol (Mol): the child molecule, ie parent + child block attached correctly.
        anchor_indices (Tuple[int, int]): the indices of parent and child anchor nodes.
    """

    # get the child block info, direcly from the vocabulary
    child_block_smiles = blocks_df["block_smi"].values[child_block_index]
    child_block_r = blocks_df["block_r"].values[child_block_index]
    child_block = Chem.MolFromSmiles(child_block_smiles)

    child_anchor_index = child_block_r[0]

    # build the new input for mol_from_frags, with only 2 fragments: the parent molecule
    # and the child block.
    new_jbonds = [[0, 1, parent_anchor_index, child_anchor_index]]
    new_frags = [parent_mol, child_block]

    child_mol, bond = mol_from_frag(jun_bonds=new_jbonds, frags=new_frags)
    anchor_indices = bond[-1]
    return child_mol, anchor_indices
