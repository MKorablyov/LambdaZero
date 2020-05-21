from LambdaZero.datasets.brutal_dock.models.model_base import get_list_of_smiles_from_batch
import LambdaZero.datasets.brutal_dock.models.hierarchical_mpnn as hmpnn


def test_get_list_of_smiles_from_batch(real_molecule_batch, realistic_smiles):
    lst = get_list_of_smiles_from_batch(real_molecule_batch)
    for smiles in lst:
        assert smiles in realistic_smiles


def test_build_vocab_from_simles_list(realistic_smiles, realistic_hgraph_vocab):
    vocab = hmpnn.build_vocab_from_simles_list(realistic_smiles)
    for v in vocab:
        assert v in realistic_hgraph_vocab


def test_feather_to_smiles_list(one_tmp_feather_file, realistic_smiles):
    smiles_list = hmpnn.feather_to_smiles_list(one_tmp_feather_file)
    for s in realistic_smiles:
        assert s in smiles_list


def test_feather_list_to_smiles_list(list_of_tmp_feather_files, realistic_smiles):
    smiles_list = hmpnn.feather_to_smiles_list(list_of_tmp_feather_files)
    for s in realistic_smiles:
        assert s in smiles_list

