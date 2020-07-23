"""
The goal of this script is to manipulate the orientation of the added block and to compute the
energy vs. angle of block.

The work is somewhat incomplete, and the energy of the original molecule is inconsistent with
the energy following a ConstrainEmbed.
"""
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem.AllChem import ConstrainedEmbed
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag, Chem
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.geometry import (
    get_center_of_mass,
    get_inertia_tensor,
    get_inertia_contribution,
    project_direction_out_of_tensor,
    rotate_points_about_axis,
)
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_atomic_masses,
    set_positions_on_conformer,
    get_mmff_force_field,
    get_mmff_energy, get_lowest_energy_and_mol_with_hydrogen,
)
from LambdaZero.examples.env3d.sanity_checks.check_atomic_order import MolecularConnection
from LambdaZero.examples.env3d.utilities import get_angles_in_degrees

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
chimera_bin = os.path.join(programs_dir, "chimera/bin/chimera")

results_dir = Path(summaries_dir).joinpath("env3d")


def get_angle_between_parent_and_child(parent_vector, child_vector, n_axis):
    x_hat = parent_vector
    y_hat = np.cross(n_axis, x_hat)

    projection_x = np.dot(child_vector, x_hat)
    projection_y = np.dot(child_vector, y_hat)

    ratio = np.abs(projection_y / projection_x)

    if projection_x >= 0.0 and projection_y >= 0.0:
        theta = np.arctan(ratio)
    elif projection_x < 0.0 and projection_y >= 0.0:
        theta = np.pi - np.arctan(ratio)
    elif projection_x < 0.0 and projection_y < 0.0:
        theta = np.pi + np.arctan(ratio)
    else:
        theta = 2 * np.pi - np.arctan(ratio)

    return theta


def get_molecular_orientation_vector_from_positions_and_masses(
    masses: np.array, positions: np.array, anchor_point: np.array, n_axis: np.array
) -> np.array:
    center_of_mass = get_center_of_mass(masses, positions)
    relative_positions = positions - center_of_mass
    inertia_cm = get_inertia_tensor(masses, relative_positions)
    total_mass = np.sum(masses)
    d = anchor_point - center_of_mass
    inertia_d = get_inertia_contribution(total_mass, d)
    total_inertia = inertia_cm + inertia_d
    return get_molecular_orientation_vector_from_inertia(total_inertia, n_axis)


def get_molecular_orientation_vector_from_inertia(total_inertia, n_axis):
    projected_inertia = project_direction_out_of_tensor(total_inertia, n_axis)
    eigs, u_matrix = np.linalg.eigh(projected_inertia)
    orientation_vector = u_matrix[:, 2]
    return orientation_vector


def plot_molecule_and_block_with_rotation_axis(
    mol: Mol, parent_size: int, anchor_indices: Tuple
):
    """
    Ugly kluge to see what the molecule looks like, as well as the orientation vectors  decorating it.
    """
    connection = MolecularConnection(mol)

    all_positions = mol.GetConformer().GetPositions()
    all_masses = get_atomic_masses(mol)

    parent_anchor_index, child_anchor_index = anchor_indices
    parent_anchor = all_positions[parent_anchor_index]
    child_anchor = all_positions[child_anchor_index]

    n_axis = child_anchor - parent_anchor
    n_axis /= np.linalg.norm(n_axis)

    parent_positions = all_positions[:parent_size]
    parent_masses = all_masses[:parent_size]
    parent_cm = get_center_of_mass(parent_masses, parent_positions)

    parent_vector = get_molecular_orientation_vector_from_positions_and_masses(
        parent_masses, parent_positions, parent_anchor, n_axis
    )

    child_positions = all_positions[parent_size:]
    child_masses = all_masses[parent_size:]
    child_cm = get_center_of_mass(child_masses, child_positions)
    child_vector = get_molecular_orientation_vector_from_positions_and_masses(
        child_masses, child_positions, child_anchor, n_axis
    )

    color_dict = {1: 'white', 6: "black", 7: "yellow", 8: "blue"}
    colors = [color_dict[a.GetAtomicNum()] for a in mol.GetAtoms()]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        all_positions[:, 0],
        all_positions[:, 1],
        all_positions[:, 2],
        "o",
        color=colors,
        s=50,
    )

    for node_index in connection.nodes:
        p1 = all_positions[node_index]
        for neighbor_index in connection.neighbors[node_index]:
            p2 = all_positions[neighbor_index]

            v = np.stack([p1, p2], axis=0)
            ax.plot(v[:, 0], v[:, 1], v[:, 2], "-", color="k")

    ax.scatter(*parent_cm, marker="x", color="red")
    ax.scatter(*child_cm, marker="x", color="green")

    ax.quiver(*parent_anchor, *parent_vector, color="red", lw=2)
    ax.quiver(*child_anchor, *child_vector, color="green", lw=2)

    ax.quiver(*parent_anchor, *n_axis, color="black", lw=1)

    x1 = parent_anchor - 3 * n_axis
    x2 = parent_anchor + 3 * n_axis
    v = np.stack([x1, x2], axis=0)
    ax.plot(v[:, 0], v[:, 1], v[:, 2], "-", color="k")

    min = np.min(all_positions.flatten())
    max = np.max(all_positions.flatten())

    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_zlim(min, max)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    return fig


number_of_blocks = 5
number_of_iterations = 100
num_conf = 25

random_seed = 1

if __name__ == "__main__":

    np.random.seed(0)

    molMDP = MolMDP(blocks_file=blocks_file)
    molMDP.reset()
    molMDP.random_walk(number_of_blocks)

    n = 2
    mol1, mol_bond1 = mol_from_frag(
        jun_bonds=molMDP.molecule.jbonds[: n - 1],
        frags=molMDP.molecule.blocks[:n],
        optimize=False,
    )

    parent = MolecularConnection(mol1)
    parent_size = mol1.GetNumAtoms()

    mol2, mol_bond2 = mol_from_frag(
        jun_bonds=molMDP.molecule.jbonds[:n],
        frags=molMDP.molecule.blocks[: n + 1],
        optimize=False,
    )
    #optimize_mol_in_place(mol2)

    min_energy, mol2, _ = get_lowest_energy_and_mol_with_hydrogen(mol2, num_conf, random_seed=random_seed)
    mol2 = Chem.RemoveHs(mol2)

    anchor_indices = (mol_bond2[-1][0], mol_bond2[-1][1])

    fig1 = plot_molecule_and_block_with_rotation_axis(mol2, parent_size, anchor_indices)
    fig1.suptitle("original orientation")

    all_positions = mol2.GetConformer().GetPositions()
    all_masses = get_atomic_masses(mol2)

    parent_anchor = all_positions[anchor_indices[0]]
    child_anchor = all_positions[anchor_indices[1]]

    n_axis = child_anchor - parent_anchor
    n_axis /= np.linalg.norm(n_axis)

    parent_positions = all_positions[:parent_size]
    parent_masses = all_masses[:parent_size]
    parent_vector = get_molecular_orientation_vector_from_positions_and_masses(
        parent_masses, parent_positions, parent_anchor, n_axis
    )

    child_positions = all_positions[parent_size:]
    child_masses = all_masses[parent_size:]
    child_vector = get_molecular_orientation_vector_from_positions_and_masses(
        child_masses, child_positions, child_anchor, n_axis
    )

    original_angle = get_angle_between_parent_and_child(
        parent_vector, child_vector, n_axis
    )
    original_angle_in_degrees = get_angles_in_degrees(original_angle)

    zero_angle_child_positions = rotate_points_about_axis(
        child_positions, child_anchor, n_axis, -original_angle
    )
    zero_angle_positions = np.copy(all_positions)
    zero_angle_positions[parent_size:] = zero_angle_child_positions

    zero_angle_mol = Mol(mol2)
    set_positions_on_conformer(zero_angle_mol, zero_angle_positions)

    fig2 = plot_molecule_and_block_with_rotation_axis(zero_angle_mol, parent_size, anchor_indices)
    fig2.suptitle("Zero angle orientation")

    list_angles = np.linspace(0, 2.0 * np.pi, 101)
    list_energies = []
    for angle in tqdm(list_angles):
        rotated_child_positions = rotate_points_about_axis(
            zero_angle_child_positions, child_anchor, n_axis, angle
        )
        rotated_positions = np.copy(zero_angle_positions)
        rotated_positions[parent_size:] = rotated_child_positions

        rotated_mol = Mol(zero_angle_mol)
        set_positions_on_conformer(rotated_mol, rotated_positions)

        Chem.SanitizeMol(rotated_mol)
        rotated_mol_with_hydrogens = Chem.AddHs(rotated_mol)
        ConstrainedEmbed(
            rotated_mol_with_hydrogens,
            rotated_mol,
            useTethers=True,
            randomseed=2342,
            getForceField=get_mmff_force_field,
        )
        energy = get_mmff_energy(rotated_mol_with_hydrogens)
        list_energies.append(energy)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        get_angles_in_degrees(list_angles),
        list_energies,
        "bo-",
        label="constrained embedding energy",
    )

    ylims = ax.set_ylim()
    xlims = ax.set_xlim()
    ax.vlines(original_angle_in_degrees, *ylims, color="r", label="original angle")
    ax.hlines(min_energy, *ylims, color="g", label="original energy")

    ax.set_xlabel("angle (degrees)")
    ax.set_ylabel("energy (kcal/mol)")
    ax.set_xlim(0, 360)
    ax.legend(loc=0)
    plt.show()
