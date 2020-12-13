import os
import pandas as pd

from rdkit import Chem

from LambdaZero.utils import get_external_dirs
datasets_dir, _, _ = get_external_dirs()
zinc20_root = os.path.join(datasets_dir, "zinc20")


def _parse_features(feature_str):
    bioactivity_class = {'unknown': 0, 'in-vitro': 1, 'in-cell': 2, 'in-vivo': 3, 'in-man': 4, 'investigational': 5, 'world': 6, 'fda': 7}
    biogenic_class = {'unknown': 0, 'np-derived': 1, 'metabolite': 2, 'endogenous': 3, 'biogenic': 4}
    is_aggregator = {'unknown': 0, 'aggregator': 1}
    bioactivity = 0
    biogenic = 0
    aggregator = 0

    if isinstance(feature_str, str):
        features = feature_str.split(',')
        for feature in features:
            bioactivity = max(bioactivity, bioactivity_class.get(feature, 0))
            biogenic = max(biogenic, biogenic_class.get(feature, 0))
            aggregator = max(aggregator, is_aggregator.get(feature, 0))
    return bioactivity, biogenic, aggregator


def _get_stats(smi):
    B = 5; C = 6; N = 7; O = 8; F = 9
    Si = 14; P = 15; S = 16; Cl = 17
    Br = 35; Sn = 50; I = 53

    mol = Chem.MolFromSmiles(smi)

    n_atoms_explicit = mol.GetNumAtoms()
    n_atoms = mol.GetNumAtoms(onlyExplicit=False)

    atoms = mol.GetAtoms()  # those are explicit only
    unique_atoms = set(atom.GetAtomicNum() for atom in atoms)
    has_B = int(B in unique_atoms)
    has_C = int(C in unique_atoms)
    has_N = int(N in unique_atoms)
    has_O = int(O in unique_atoms)
    has_F = int(F in unique_atoms)
    has_Si = int(Si in unique_atoms)
    has_P = int(P in unique_atoms)
    has_S = int(S in unique_atoms)
    has_Cl = int(Cl in unique_atoms)
    has_Br = int(Br in unique_atoms)
    has_Sn = int(Sn in unique_atoms)
    has_I = int(I in unique_atoms)

    return n_atoms_explicit, n_atoms, has_B, has_C, has_N, has_O, has_F, has_Si, has_P, has_S, has_Cl, has_Br, has_Sn, has_I


if __name__ == "__main__":
    zinc20_txt = os.path.join(zinc20_root, "zinc20_2D_txt")
    files = [os.path.join(zinc20_txt, file) for file in os.listdir(zinc20_txt)]

    data = pd.concat([pd.read_table(file) for file in files])
    data.reset_index(drop=True, inplace=True)
    data.drop(columns=['reactive', 'purchasable', 'tranche_name'], inplace=True)   # we only have anodyne, and in-stock+agent
    data.zinc_id = data.zinc_id.apply(lambda idx: int(idx[4:]))                    # original ZINC id = "ZINC" + 12 digits
    data.rename(columns={'zinc_id': 'int_zinc_id'}, inplace=True)
    data[['bioactivity_class', 'biogenic_class', 'is_aggregator']] = pd.DataFrame(data.features.apply(lambda f_str: _parse_features(f_str)).tolist(), index=data.index)
    data.drop(columns=['features'], inplace=True)

    data[['n_atoms_explicit', 'n_atoms',
          'has_B', 'has_C', 'has_N', 'has_O', 'has_F',
          'has_Si', 'has_P', 'has_S', 'has_Cl',
          'has_Br', 'has_Sn', 'has_I']] = pd.DataFrame(data.smiles.apply(lambda smi: _get_stats(smi)).tolist(), index=data.index)

    data.to_csv(os.path.join(zinc20_root, "zinc20_base.csv"), index=False)  # index does not carry any info, .feather inflates memory usage
    print(len(data))
