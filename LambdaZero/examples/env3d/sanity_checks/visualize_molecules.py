"""
The goal of this script is to manipulate the orientation of the added block and to compute the
energy vs. angle of block.

The work is somewhat incomplete, and the energy of the original molecule is inconsistent with
the energy following a ConstrainEmbed.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdmolfiles import MolFromSmiles

import LambdaZero.utils
from LambdaZero.examples.env3d.geometry import get_angle_between_parent_and_child
from LambdaZero.examples.env3d.rdkit_utilities import (
    get_lowest_energy_and_mol_with_conformer, get_mmff_energy, )
from LambdaZero.examples.env3d.sanity_checks.molecule_plotting import plot_molecule_and_block_with_rotation_axis
from LambdaZero.examples.env3d.sanity_checks.utils import get_child_molecule, extract_mol_geometry
from LambdaZero.examples.env3d.utilities import get_angles_in_degrees

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
blocks_file = os.path.join(datasets_dir, "fragdb/blocks_PDB_105.json")

results_dir = Path(summaries_dir).joinpath("env3d")

number_of_blocks = 10
number_of_iterations = 100
num_conf = 10

random_seed = 0


dataset_path = Path("/Users/bruno/LambdaZero/summaries/env3d/dataset/from_cluster/RUN5/data/raw/combined_dataset.feather")

if __name__ == "__main__":
    blocks_df = pd.read_json(blocks_file)

    df = pd.read_feather(dataset_path)

    data_row = df.loc[0]

    parent_smiles = data_row['smi']
    flat_coord = np.frombuffer(data_row["coord"])
    parent_positions = flat_coord.reshape(len(flat_coord) // 3, 3)

    parent_mol = MolFromSmiles(parent_smiles)
    parent_size = parent_mol.GetNumAtoms()

    min_parent_energy, parent_mol_with_hydrogens, _ = get_lowest_energy_and_mol_with_conformer(parent_mol,
                                                                                               num_conf,
                                                                                               random_seed=random_seed)

    id = parent_mol_with_hydrogens.GetConformer().GetId()
    computed_parent_energy = get_mmff_energy(parent_mol_with_hydrogens, conf_id=id)

    child_block_index = data_row['attachment_block_class']
    parent_anchor_index = data_row['attachment_node_idx']

    child_mol, anchor_indices = get_child_molecule(parent_mol, child_block_index, parent_anchor_index, blocks_df)


    min_child_energy, child_mol_with_hydrogens, _ = get_lowest_energy_and_mol_with_conformer(child_mol,
                                                                                               num_conf,
                                                                                               random_seed=random_seed)

    id = child_mol_with_hydrogens.GetConformer().GetId()
    computed_child_energy = get_mmff_energy(child_mol_with_hydrogens, conf_id=id)

    relaxed_child_mol = Chem.RemoveHs(child_mol_with_hydrogens)

    geometry_dict = extract_mol_geometry(anchor_indices, relaxed_child_mol, parent_size)

    expected_angle = data_row['attachment_angle']

    expected_angle_in_degrees = get_angles_in_degrees(expected_angle)

    computed_angle = get_angle_between_parent_and_child(geometry_dict['parent_vector'],
                                                        geometry_dict['child_vector'],
                                                        geometry_dict['n_axis'])

    fig1 = plot_molecule_and_block_with_rotation_axis(relaxed_child_mol, parent_size, anchor_indices)
    fig1.suptitle("original orientation")
