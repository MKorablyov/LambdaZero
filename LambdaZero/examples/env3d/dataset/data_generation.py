import copy
from typing import Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles
from tqdm import tqdm

from LambdaZero.chem import mol_from_frag
from LambdaZero.environments import MolMDP
from LambdaZero.examples.env3d.geometry import (
    get_positions_aligned_with_parent_inertia_tensor,
    get_n_axis_and_angle,
)
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_lowest_energy_and_mol_with_hydrogen,
    get_atomic_masses,
    get_index_permutation_between_equivalent_molecules,
)


def extract_lowest_energy_child(
    reference_molMDP: MolMDP,
    attachment_stem_idx: int,
    num_conf: int,
    max_iters: int,
    random_seed: int,
):
    """
    This method extracts the lowest energy child conformer molecule given a reference MDP object which
    already contains the parent molecule as its state, and provided the attachment information.

    The possible blocks that can be attached are defined by the reference MDP.

    The algorithm generates a child molecule for every possible block, embeds and relax the child
    molecules to obtain a conformer and an energy, and then selects the child molecule with lowest energy.

    Args:
        reference_molMDP (molMDP): MDP object where a parent molecule is present.
        attachment_stem_idx (int): attachment_stem id indicating where the child block will be attached.
        num_conf (int): number of configurations attempted to embed the full child molecule as a conformer
        max_iters (int): number of iterations to converge the atomic positions
        random_seed (int): random seed used to embed molecule and create a conformer.

    Returns:

        relaxed_mol (Mol): child molecule, with relaxed positions conformer,
        block_idx (int): index of the lowest energy block that was attached
        anchor_indices (Tuple): atomic indices of the (parent, child block) attachment atoms

    """

    list_block_indices = np.arange(reference_molMDP.num_blocks)

    list_min_energy = []
    list_relaxed_mol_with_hydrogen = []
    list_bond = []

    for block_idx in tqdm(list_block_indices):
        molMDP = copy.deepcopy(reference_molMDP)
        molMDP.add_block(block_idx=block_idx, stem_idx=attachment_stem_idx)
        mol, bond = mol_from_frag(
            jun_bonds=molMDP.molecule.jbonds, frags=molMDP.molecule.blocks
        )

        try:
            min_energy, mol_with_hydrogens, _ = get_lowest_energy_and_mol_with_hydrogen(
                mol, num_conf, max_iters=max_iters, random_seed=random_seed
            )
        except ValueError:
            min_energy = np.NaN
            mol_with_hydrogens = np.NaN

        list_min_energy.append(min_energy)
        list_relaxed_mol_with_hydrogen.append(mol_with_hydrogens)
        list_bond.append(bond)

    min_index = int(np.nanargmin(list_min_energy))

    block_idx = list_block_indices[min_index]
    bond = list_bond[min_index]
    anchor_indices = (bond[-1][0], bond[-1][1])

    relaxed_mol = Chem.RemoveHs(list_relaxed_mol_with_hydrogen[min_index])

    return relaxed_mol, block_idx, anchor_indices


def get_data_row(
    reference_molMDP: MolMDP,
    attachment_stem_idx: int,
    num_conf: int,
    max_iters: int,
    random_seed: int,
):
    """

    This method is a driver that extracts the lowest energy child conformer molecule
    by calling `extract_lowest_energy_child` and then does the needed book keeping to
    create a row of data to be appended to the pandas dataframe to be written as a feather file.


    Args:
        reference_molMDP (molMDP): MDP object where a parent molecule is present.
        attachment_stem_idx (int): attachment_stem id indicating where the child block will be attached.
        num_conf (int): number of configurations attempted to embed the full child molecule as a conformer
        max_iters (int): number of iterations to converge the atomic positions
        random_seed (int): random seed used to embed molecule and create a conformer.


    Returns:
        data_row (dict): a dictionary contraining a row of data to be saved as a feather file.

    """

    parent_mol = reference_molMDP.molecule.mol
    number_of_parent_atoms = parent_mol.GetNumAtoms()

    relaxed_mol, block_idx, anchor_indices = extract_lowest_energy_child(
        reference_molMDP, attachment_stem_idx, num_conf, max_iters, random_seed
    )
    attachment_index = anchor_indices[0]

    all_unnormalized_positions = relaxed_mol.GetConformer().GetPositions()
    all_masses = get_atomic_masses(relaxed_mol)

    all_positions = get_positions_aligned_with_parent_inertia_tensor(
        all_unnormalized_positions, all_masses, number_of_parent_atoms
    )

    n_axis, angle_in_radian = get_n_axis_and_angle(
        all_positions, all_masses, anchor_indices, number_of_parent_atoms
    )

    parent_positions = all_positions[:number_of_parent_atoms]

    # the atomic order in parent_mol is NOT THE SAME as the atomic order after creating the Mol from
    # a smiles string. This method ensures that the positions and attachment index are consistent
    # with the Mol created from a smiles.
    parent_smiles, permuted_positions, permuted_attachment_index = get_smiles_and_consistent_positions(
        parent_mol, parent_positions, attachment_index
    )

    return {
        "smi": parent_smiles,
        "coord": permuted_positions,
        "n_axis": n_axis,
        "attachment_node_index": permuted_attachment_index,
        "attachment_angle": angle_in_radian,
        "attachment_block_index": block_idx,
    }


def get_smiles_and_consistent_positions(
    mol: Mol, positions: np.array, attachment_index: int
) -> Tuple[str, np.array, int]:
    """
    This method creates the smiles corresponding to the input Mol and a permutation of the
    input positions and attachment index that are consistent with the smiles atomic ordering.

    Args:
        mol (Mol): a molecule in Mol form
        positions (np.array): atomic positions corresponding to the atoms in mol
        attachment_index (int): the index of one of these atoms, which is used to attach a child node.

    Returns:
        smiles (str): the smiles string corresponding to mol
        smiles_consistent_positions (np.array): permuted positions consistent with the atom ordering implicit in smiles
        smiles_consistent_attachment_index (int): correct index after permutation

    """

    smiles = MolToSmiles(mol)
    smiles_mol = MolFromSmiles(smiles)
    if smiles_mol is None:
        raise ValueError("MolFromSmiles failed to generate a Mol. Bad molecule")

    permutation = get_index_permutation_between_equivalent_molecules(mol, smiles_mol)

    # permutation is an array of indices such that, for p_i = permutation[i],
    #  the atom with index "i" in smiles_mol is equivalent to atom with index "p_i" in mol.
    permuted_positions = positions[permutation]

    # To find the index of the attachment atom, we have that p_i = attachment_index; we have to
    # find the index "i" in smiles_mol that corresponds to this.
    permuted_attachment_index = list(permutation).index(attachment_index)

    return smiles, permuted_positions, permuted_attachment_index
