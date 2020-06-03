
from rdkit.Chem import AllChem


def morgan_fp_from_smiles(smiles_str, radius=2, number_bits=1024):
    mol = AllChem.MolFromSmiles(smiles_str)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=number_bits)


