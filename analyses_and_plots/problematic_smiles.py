import pandas as pd
from rdkit.Chem import Draw
from rdkit.Chem.rdmolfiles import MolFromSmiles

from LambdaZero.chem import mol_to_graph, build_mol, mpnn_feat
from analyses_and_plots.chemprop_adaptor import graph_to_mol
from analyses_and_plots import ANALYSIS_RESULTS_DIR
from analyses_and_plots.analysis_utils import get_number_of_hydrogens, get_atom_series

smiles = "Nn1cnc2c([PH](=O)[O-])ncnc21"
problematic_atom_index = 8  # I know this by inspection

molecule_images_path = ANALYSIS_RESULTS_DIR.joinpath("molecules_images.jpg")

if __name__ == '__main__':

    print(f"Molecule SMILE: {smiles}")

    # This is the "canonical" mol obtained directly from the smiles.
    mol_from_smiles = MolFromSmiles(smiles)
    atomic_numbers = [atom.GetAtomicNum() for atom in mol_from_smiles.GetAtoms()]
    print(f"Atomic numbers : {atomic_numbers}")
    print(f"\n")

    h_counts_from_smiles = get_number_of_hydrogens(mol_from_smiles)
    print(f"hydrogen count for canonical mol:\t {h_counts_from_smiles}")

    # mol built using internal code from LambdaZero
    mol_from_build_mol = build_mol(smiles=smiles)[0]
    h_counts_from_build_mol = get_number_of_hydrogens(mol_from_build_mol)
    print(f"hydrogen count for LambdaZero mol:\t {h_counts_from_build_mol}")

    atmfeat, coord, bond, bondfeat = mpnn_feat(mol_from_build_mol, ifcoord=True, panda_fmt=True)
    print(f"hydrogen count, LambdaZero extract:\t {list(atmfeat['num_hs'].values)}")

    # Extract the pytorch_geometric graph for this molecule.
    molecule_graph = mol_to_graph(smiles)

    # Best guess at what the mol object is, starting from the pytorch_geometric object.
    mol_from_adaptor = graph_to_mol(molecule_graph)
    h_counts_from_adaptor = get_number_of_hydrogens(mol_from_adaptor)
    print(f"hydrogen count for my adaptor mol:\t {h_counts_from_adaptor}")

    print(f" -----> The adaptor gets the hydrogen count wrong at index {problematic_atom_index} <------")

    problematic_atom_from_smiles = mol_from_smiles.GetAtoms()[problematic_atom_index]
    problematic_atom_from_adaptor = mol_from_adaptor.GetAtoms()[problematic_atom_index]

    expected_series = get_atom_series(problematic_atom_from_smiles)
    expected_series.name = 'from SMILES'
    computed_series = get_atom_series(problematic_atom_from_adaptor)
    computed_series.name = 'from adaptor'
    df = pd.concat([expected_series, computed_series], axis=1)

    print(f"\nProperties of atom {problematic_atom_index}")
    print(df)

    output = Draw.MolsToGridImage([mol_from_smiles, mol_from_build_mol, mol_from_adaptor],
                                  legends=['from smiles', 'from build mol', 'from adaptor'])

    output.save(molecule_images_path)

    print(f"molecule images saved at {molecule_images_path}")
