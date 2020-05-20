from LambdaZero.datasets.brutal_dock.models.model_base import get_list_of_smiles_from_batch
import LambdaZero.datasets.brutal_dock.models.hierarchical_mpnn as hmpnn


def test_get_list_of_smiles_from_batch(real_molecule_batch, realistic_smiles):
    lst = get_list_of_smiles_from_batch(real_molecule_batch)
    for smiles in lst:
        assert smiles in realistic_smiles

