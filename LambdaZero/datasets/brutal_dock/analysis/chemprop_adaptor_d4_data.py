"""
This script applies the chemprop adaptor to try to reconstruct the RDKit Mol object from the
pytorch_geometric graph data, itself generated from a smiles string. At this time, this is
conversion fails about half the time, and it appears hopeless to reconstruct quality Mol objects
without a deep understanding of how RDKit works.
"""

import numpy as np

import pandas as pd
from rdkit.Chem.rdchem import AtomValenceException, KekulizeException
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles

from LambdaZero.chem import mol_to_graph
from LambdaZero.datasets.brutal_dock import BRUTAL_DOCK_DATA_DIR
from LambdaZero.datasets.brutal_dock.analysis.analysis_utils import get_all_formal_charges, \
    are_two_mols_the_same_for_chemprop
from LambdaZero.datasets.brutal_dock.models.chemprop_adaptor import graph_to_mol

feather_filenames = ["dock_blocks105_walk40_clust.feather", "dock_blocks105_walk40_2_clust.feather"]

number_of_molecules_to_run = 100

if __name__ == "__main__":

    list_df = []
    for raw_file_name in feather_filenames:
        d4_feather_data_path = BRUTAL_DOCK_DATA_DIR.joinpath(f"d4/raw/{raw_file_name}")
        df = pd.read_feather(d4_feather_data_path)
        list_df.append(df)

    raw_data_df = pd.concat(list_df).reset_index(drop=True)

    list_smiles = list(raw_data_df['smiles'].values)

    count_problematic_conversion = 0
    count_correct_conversion = 0
    count_wrong_conversion = 0
    for raw_smiles in list_smiles[:number_of_molecules_to_run]:

        smiles = MolToSmiles(MolFromSmiles(raw_smiles), canonical=True)
        assert smiles == raw_smiles

        mol_from_smiles = MolFromSmiles(smiles)

        molecule_graph = mol_to_graph(smiles)

        try:
            mol_from_adaptor = graph_to_mol(molecule_graph)
        except (AtomValenceException, KekulizeException):
            print(f"-----------------------------------------")
            print(f"problem with {smiles}")
            count_problematic_conversion += 1
            continue

        same_for_chemprop = are_two_mols_the_same_for_chemprop(mol_from_smiles, mol_from_adaptor)
        if same_for_chemprop:
            count_correct_conversion += 1
        else:
            count_wrong_conversion += 1

        reconstructed_smiles = MolToSmiles(mol_from_adaptor, canonical=True)

        formal_charges = get_all_formal_charges(mol_from_smiles)
        total_charge = np.sum(formal_charges)
        total_abs_charge = np.sum(np.abs(formal_charges))

        if not same_for_chemprop:
            print(f"-----------------------------------------")
            print(f"original smiles = {smiles}")
            print(f"recon. smiles   = {reconstructed_smiles}")
            print(f"Total charge    = {total_charge}")
            print(f"sum abs(charges)= {total_abs_charge}")

    print(f"=========================================")
    print(f"Final Statistics")
    print(f"number of sampled smiles      = {number_of_molecules_to_run}")
    print(f"number of broken conversions  = {count_problematic_conversion}")
    print(f"number of wrong conversions   = {count_wrong_conversion}")
    print(f"number of correct conversions = {count_correct_conversion}")
