"""
The goal of this script is to manipulate the orientation of the added block and to compute the
energy vs. angle of block.

The work is somewhat incomplete, and the energy of the original molecule is inconsistent with
the energy following a ConstrainEmbed.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem.AllChem import ConstrainedEmbed
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag, Chem
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.geometry import (
    rotate_points_about_axis, get_angle_between_parent_and_child,
    get_molecular_orientation_vector_from_positions_and_masses,
)
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_atomic_masses,
    set_positions_on_conformer,
    get_mmff_force_field,
    get_mmff_energy, get_lowest_energy_and_mol_with_conformer,
)
from LambdaZero.examples.env3d.molecular_connection import MolecularConnection
from LambdaZero.examples.env3d.sanity_checks.molecule_plotting import plot_molecule_and_block_with_rotation_axis
from LambdaZero.examples.env3d.utilities import get_angles_in_degrees

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
chimera_bin = os.path.join(programs_dir, "chimera/bin/chimera")

results_dir = Path(summaries_dir).joinpath("env3d")

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

    min_energy, mol2, _ = get_lowest_energy_and_mol_with_conformer(mol2, num_conf, random_seed=random_seed)
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
