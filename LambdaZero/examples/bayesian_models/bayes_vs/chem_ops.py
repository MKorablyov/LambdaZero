
import functools

from rdkit.Chem import AllChem
import numpy as np

@functools.lru_cache(int(1e6))
def morgan_fp_from_smiles(smiles_str, radius=2, number_bits=1024):
    mol = AllChem.MolFromSmiles(smiles_str)
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=number_bits), dtype=np.float32)


