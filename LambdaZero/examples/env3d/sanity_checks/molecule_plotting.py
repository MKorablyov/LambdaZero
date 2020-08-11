from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from rdkit.Chem.rdchem import Mol

from LambdaZero.examples.env3d.molecular_connection import MolecularConnection
from LambdaZero.examples.env3d.sanity_checks.utils import extract_mol_geometry

# standard atomic colors in molecules https://en.wikipedia.org/wiki/CPK_coloring
CPK_COLOR_SCHEME = {
    1: "white",  # hydrogen
    6: "black",  # carbon
    7: "blue",  # nitrogen
    8: "red",  # oxygen
    9: "green",  # fluorine
    15: "orange",  # phosphorus
    16: "yellow",  # sulfur
    17: "green",  # chlorine
    35: "darkred",  # bromine
    53: "darkviolet",  # iodine
}

DEFAULT_COLOR = "pink"


def plot_molecule_and_block_with_rotation_axis(
    mol: Mol, parent_size: int, anchor_indices: Tuple
):
    """
    This method plots the molecule in 3D and identifies various landmarks like centers of mass,
    axis of rotation, directions, etc.

    Args:
        mol (Mol): the molecule to be plotted, assumed to contain a conformer
        parent_size (int): number of parent atoms
        anchor_indices (Tuple[int, int]): indices of the parent and child atoms that bind the child block to the parent molecule

    Returns:
        fig (matplotlib.figure.Figure): the figure object with the moleculed plotted on it.
    """

    geometry_dict = extract_mol_geometry(anchor_indices, mol, parent_size)

    # we will use all_positions again and again; it's worth defining its own internal variable
    all_positions = geometry_dict["all_positions"]

    colors = [
        CPK_COLOR_SCHEME.get(a.GetAtomicNum(), DEFAULT_COLOR) for a in mol.GetAtoms()
    ]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        all_positions[:, 0],
        all_positions[:, 1],
        all_positions[:, 2],
        "o",
        color=colors,
        s=100,
    )

    connection = MolecularConnection(mol)
    for node_index in connection.nodes:
        p1 = all_positions[node_index]
        for neighbor_index in connection.neighbors[node_index]:
            p2 = all_positions[neighbor_index]

            v = np.stack([p1, p2], axis=0)
            ax.plot(v[:, 0], v[:, 1], v[:, 2], "-", color="k")

    ax.scatter(*geometry_dict["parent_cm"], marker="x", color="red", label="parent CM")
    ax.scatter(*geometry_dict["child_cm"], marker="x", color="green", label="child CM")

    ax.quiver(
        *geometry_dict["parent_anchor"],
        *geometry_dict["parent_vector"],
        color="red",
        lw=2,
        label="parent direction"
    )
    ax.quiver(
        *geometry_dict["child_anchor"],
        *geometry_dict["child_vector"],
        color="green",
        lw=2,
        label="child direction"
    )
    ax.quiver(
        *geometry_dict["parent_anchor"], *geometry_dict["n_axis"], color="purple", lw=2
    )

    a = geometry_dict["parent_anchor"]
    b = geometry_dict["n_axis"]
    x1 = a - 3 * b
    x2 = a + 3 * b

    v = np.stack([x1, x2], axis=0)
    ax.plot(v[:, 0], v[:, 1], v[:, 2], "-", color="purple", label="rotation axis")

    position_min = np.min(all_positions.flatten())
    position_max = np.max(all_positions.flatten())

    ax.set_xlim(position_min, position_max)
    ax.set_ylim(position_min, position_max)
    ax.set_zlim(position_min, position_max)

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.legend(loc=0)

    return fig
