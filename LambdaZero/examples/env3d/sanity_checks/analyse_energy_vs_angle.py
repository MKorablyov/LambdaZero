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
from LambdaZero.examples.env3d.geometry import get_angle_between_parent_and_child, rotate_points_about_axis
from LambdaZero.examples.env3d.molecular_connection import MolecularConnection
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_mmff_force_field,
    get_mmff_energy, get_lowest_energy_and_mol_with_conformer, set_positions_on_conformer,
)
from LambdaZero.examples.env3d.sanity_checks.molecule_plotting import plot_molecule_and_block_with_rotation_axis
from LambdaZero.examples.env3d.sanity_checks.utils import extract_mol_geometry
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

    n = 4
    mol1, mol_bond1 = mol_from_frag(
        jun_bonds=molMDP.molecule.jbonds[: n - 1],
        frags=molMDP.molecule.blocks[:n],
        optimize=False,
    )

    parent = MolecularConnection(mol1)
    parent_size = mol1.GetNumAtoms()

    mol_with_hydrogen, mol_bond2 = mol_from_frag(
        jun_bonds=molMDP.molecule.jbonds[:n],
        frags=molMDP.molecule.blocks[: n + 1],
        optimize=False,
    )

    min_energy, mol2, _ = get_lowest_energy_and_mol_with_conformer(mol_with_hydrogen, num_conf, random_seed=random_seed)
    mol2 = Chem.RemoveHs(mol2)

    anchor_indices = (mol_bond2[-1][0], mol_bond2[-1][1])

    fig1 = plot_molecule_and_block_with_rotation_axis(mol2, parent_size, anchor_indices)
    fig1.suptitle("original orientation")

    geometry_dict = extract_mol_geometry(anchor_indices, mol2, parent_size)

    all_positions = geometry_dict['all_positions']
    n_axis = geometry_dict['n_axis']

    parent_vector = geometry_dict['parent_vector']

    child_vector = geometry_dict['child_vector']
    child_positions = geometry_dict['child_positions']
    child_anchor = geometry_dict['child_anchor']

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
    list_rms = []
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
            randomseed=2342,
            getForceField=get_mmff_force_field,
        )
        energy = get_mmff_energy(rotated_mol_with_hydrogens)
        list_energies.append(energy)
        rms = float(rotated_mol_with_hydrogens.GetProp('EmbedRMS'))
        list_rms.append(rms)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    angles_in_degrees = get_angles_in_degrees(list_angles)
    ax.plot(angles_in_degrees,
        list_energies,
        "bo-",
        label="constrained embedding energy",
    )

    ymin, ymax = ax.set_ylim()
    ymin = min(ymin, min_energy-2)

    ax.vlines(original_angle_in_degrees, ymin, ymax, color="r", label="original angle")
    ax.hlines(min_energy, 0., 360., color="g", label="original energy")

    ax.set_ylabel("energy (kcal/mol)")
    ax.set_xlim(0, 360)
    ax.set_ylim(ymin, ymax)
    ax.legend(loc=0)

    ax2.plot(angles_in_degrees, list_rms, 'ro-')
    ax2.set_xlim(0, 360)
    ax2.set_ylabel("RMS")
    ax2.set_xlabel("angle (degrees)")

    plt.show()
