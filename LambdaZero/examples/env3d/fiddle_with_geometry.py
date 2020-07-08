"""
The goal of this script is to verify that atoms are in the same order as multiple blocks are added to
the molecule.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import LambdaZero.utils
from LambdaZero.chem import mol_from_frag
from LambdaZero.environments.molMDP import MolMDP
from LambdaZero.examples.env3d.check_atomic_order import MolecularConnection
from LambdaZero.examples.env3d.geometry import get_center_of_mass, get_inertia_tensor, get_inertia_contribution, \
    project_direction_out_of_tensor
from LambdaZero.examples.env3d.rdkit_utilities import optimize_mol_in_place, get_atomic_masses


datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")
chimera_bin = os.path.join(programs_dir, "chimera/bin/chimera")

results_dir = Path(summaries_dir).joinpath("env3d")


number_of_blocks = 5
number_of_iterations = 100

if __name__ == "__main__":

    np.random.seed(0)

    molMDP = MolMDP(blocks_file=blocks_file)
    molMDP.reset()
    molMDP.random_walk(number_of_blocks)

    n = 4
    mol1, mol_bond1 = mol_from_frag(jun_bonds=molMDP.molecule.jbonds[:n-1],
                            frags=molMDP.molecule.blocks[:n],
                            optimize=False)

    parent = MolecularConnection(mol1)
    parent_size = mol1.GetNumAtoms()

    mol2, mol_bond2 = mol_from_frag(jun_bonds=molMDP.molecule.jbonds[:n],
                            frags=molMDP.molecule.blocks[:n+1],
                            optimize=False)
    optimize_mol_in_place(mol2)

    child = MolecularConnection(mol2)

    all_positions = mol2.GetConformer().GetPositions()
    all_masses = get_atomic_masses(mol2)

    parent_positions = all_positions[:parent_size]
    parent_masses = all_masses[:parent_size]
    parent_cm = get_center_of_mass(parent_masses, parent_positions)
    relative_parent_positions = parent_positions - parent_cm
    parent_inertia_cm = get_inertia_tensor(parent_masses, relative_parent_positions)

    child_positions = all_positions[parent_size:]
    child_masses = all_masses[parent_size:]
    child_cm = get_center_of_mass(child_masses, child_positions)
    relative_child_positions = child_positions - child_cm
    child_inertia_cm = get_inertia_tensor(parent_masses, relative_parent_positions)

    parent_anchor = all_positions[mol_bond2[-1][0]]
    child_anchor = all_positions[mol_bond2[-1][1]]
    n_axis = child_anchor-parent_anchor
    n_axis /= np.linalg.norm(n_axis)

    total_parent_mass = np.sum(parent_masses)
    parent_d = parent_anchor - parent_cm
    parent_inertia_d = get_inertia_contribution(total_parent_mass, parent_d)
    parent_total_inertia = parent_inertia_cm + parent_inertia_d

    projected_parent_inertia = project_direction_out_of_tensor(parent_total_inertia, n_axis)
    eigs, u_matrix = np.linalg.eigh(projected_parent_inertia)
    parent_vector = u_matrix[:, 2]

    total_child_mass = np.sum(child_masses)
    child_d = child_anchor - child_cm
    child_inertia_d = get_inertia_contribution(total_child_mass, child_d)
    child_total_inertia = child_inertia_cm + child_inertia_d
    projected_child_inertia = project_direction_out_of_tensor(child_total_inertia, n_axis)
    eigs, u_matrix = np.linalg.eigh(projected_child_inertia)
    child_vector = u_matrix[:, 2]


    # Ugly kluge to see what the molecule looks like, as well as vectors

    color_dict = {6: 'black', 7: 'yellow', 8: 'blue'}
    colors = [color_dict[a.GetAtomicNum()] for a in mol2.GetAtoms()]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], 'o', color=colors, s=50)

    for node_index in child.nodes:
        p1 = all_positions[node_index]
        for neighbor_index in child.neighbors[node_index]:
            p2 = all_positions[neighbor_index]

            v = np.stack([p1, p2], axis=0)
            ax.plot(v[:, 0], v[:, 1], v[:, 2], '-', color='k')

    ax.scatter(*parent_cm, marker='x', color='red')
    ax.scatter(*child_cm, marker='x', color='green')

    ax.quiver(*parent_anchor, *parent_vector, color='red', lw=2)
    ax.quiver(*child_anchor, *child_vector, color='green', lw=2)

    ax.quiver(*parent_anchor, *n_axis, color='black', lw=1)

    x1 = parent_anchor - 3*n_axis
    x2 = parent_anchor + 3*n_axis
    v = np.stack([x1, x2], axis=0)
    ax.plot(v[:, 0], v[:, 1], v[:, 2], '-', color='k')

    min = np.min(all_positions.flatten())
    max = np.max(all_positions.flatten())

    ax.set_xlim(min, max)
    ax.set_ylim(min, max)
    ax.set_zlim(min, max)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
