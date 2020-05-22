from pathlib import Path

import pytest

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


def test_vocab_to_txt_file(root_dir, realistic_hgraph_vocab):
    save_filename = str(Path(root_dir).joinpath("tmp.txt"))
    hmpnn.vocab_to_txt_file(realistic_hgraph_vocab, save_filename)
    loaded_vocab = hmpnn.load_vocab_from_txt_file(save_filename)
    for v in realistic_hgraph_vocab:
        assert v in loaded_vocab

@pytest.mark.skip(reason="the hmpnn module uses `.cuda()` which cannot work on cpu only machines so we cannot test it")
def test_HMPNN_forward(real_molecule_batch, one_tmp_feather_file):
    """
    Smoke test that HMPNN runs on expected input shape.
    """
    parameters = {
        'vocab_file': one_tmp_feather_file,
        'rnn_type': "LSTM",
        'embed_size': 12,
        'hidden_size': 12,
        'depthT': 20,
        'depthG': 20,
        'dropout': 0.0,
    }
    net = hmpnn.HierarchMPNN(**parameters)
    _ = net.forward(real_molecule_batch)