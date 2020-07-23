"""
The goal of this script is to generate the dataset of molecules embedded in 3D space.
"""
import copy
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from rdkit import Chem
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.dataset.io_utilities import (
    process_row_for_writing_to_feather,
    create_or_append_feather_file,
)
from LambdaZero.examples.env3d.geometry import get_center_of_mass, get_inertia_tensor
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_lowest_energy_and_mol_with_hydrogen,
    get_atomic_masses,
)
from LambdaZero.examples.env3d.sanity_checks.analyse_energy_vs_angle import (
    get_angle_between_parent_and_child,
    get_molecular_orientation_vector_from_positions_and_masses,
)


def extract_lowest_energy_child(
    reference_molMDP: MolMDP,
    attachment_stem_idx: int,
    num_conf: int,
    max_iters: int,
    random_seed: int,
):

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


def get_positions_aligned_with_parent_inertia_tensor(
    all_positions: np.array, all_masses: np.array, number_of_parent_atoms: int
) -> np.array:

    parent_positions = all_positions[:number_of_parent_atoms]
    parent_masses = all_masses[:number_of_parent_atoms]

    parent_center_of_mass = get_center_of_mass(parent_masses, parent_positions)
    parent_inertia_cm = get_inertia_tensor(
        parent_masses, parent_positions - parent_center_of_mass
    )

    inertia_eigenvalues, u_matrix = np.linalg.eigh(parent_inertia_cm)

    normalized_positions = np.dot(all_positions - parent_center_of_mass, u_matrix)

    return normalized_positions


def get_n_axis_and_angle(
    all_positions: np.array,
    all_masses: np.array,
    anchor_indices: Tuple,
    number_of_parent_atoms: int,
):
    parent_positions = all_positions[:number_of_parent_atoms]
    parent_masses = all_masses[:number_of_parent_atoms]

    child_positions = all_positions[number_of_parent_atoms:]
    child_masses = all_masses[number_of_parent_atoms:]

    parent_anchor = all_positions[anchor_indices[0]]
    child_anchor = all_positions[anchor_indices[1]]

    n_axis = child_anchor - parent_anchor
    n_axis /= np.linalg.norm(n_axis)

    parent_vector = get_molecular_orientation_vector_from_positions_and_masses(
        parent_masses, parent_positions, parent_anchor, n_axis
    )

    child_vector = get_molecular_orientation_vector_from_positions_and_masses(
        child_masses, child_positions, child_anchor, n_axis
    )

    angle_in_radian = get_angle_between_parent_and_child(
        parent_vector, child_vector, n_axis
    )

    return n_axis, angle_in_radian


def get_data_row(
    reference_molMDP: MolMDP,
    attachment_stem_idx: int,
    num_conf: int,
    max_iters: int,
    random_seed: int,
):

    number_of_parent_atoms = reference_molMDP.molecule.mol.GetNumAtoms()

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

    return {
        "coord": parent_positions,
        "n_axis": n_axis,
        "attachment_node_index": attachment_index,
        "attachment_angle": angle_in_radian,
        "attachment_block_index": block_idx,
    }


datasets_dir, _, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
results_dir = Path(summaries_dir).joinpath("env3d/dataset/")

results_dir.mkdir(exist_ok=True)

number_of_parent_blocks = 5

output_path = results_dir.joinpath(
    f"env3d_dataset_{number_of_parent_blocks}_parent_blocks.feather"
)

num_conf = 25
max_iters = 200
random_seed = 12312

max_number_of_molecules = 1000

if __name__ == "__main__":
    np.random.seed(random_seed)

    reference_molMDP = MolMDP(blocks_file=blocks_file)

    for index in range(max_number_of_molecules):
        print(f" Computing molecule {index} of {max_number_of_molecules}")
        reference_molMDP.reset()
        reference_molMDP.random_walk(number_of_parent_blocks)
        number_of_stems = len(reference_molMDP.molecule.stems)

        if number_of_stems < 1:
            print("no stems! Moving on to next molecule")
            continue

        attachment_stem_idx = np.random.choice(number_of_stems)

        try:
            row = get_data_row(
                reference_molMDP, attachment_stem_idx, num_conf, max_iters, random_seed
            )
        except:
            print("Something went wrong while relaxing molecule. Moving on to next molecule")
            continue

        byte_row = process_row_for_writing_to_feather(row)
        create_or_append_feather_file(output_path, byte_row)
