import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from hgraph import HierMPNEncoder, MolGraph, common_atom_vocab
from LambdaZero.datasets.brutal_dock.models.model_base import ModelBase, get_list_of_smiles_from_batch


def build_vocab_from_simles_list(list_of_all_smiles):
    """
    returns a list of (smiles, smiles) tuples. This function was adapted from
    hgraph's `get_vocab.py` script, that must be ran before using hgraph
    """
    vocab = set() # turned to a list
    ## TODO: this could be parallelized
    for s in list_of_smiles:
        hmol = MolGraph(s)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    vocab = list(vocab)
    return vocab


def feather_to_smiles_list(path_or_paths_list):
    try:
        df = pd.read_feather(path_or_paths_list)
    except TypeError:
        list_df = [pd.read_feather(path) for path in path_or_paths_list]
        df = pd.concat(list_df).reset_index(drop=True)
    return df["smiles"].tolist()


def vocab_to_txt_file(vocab, filename):
    """ Saves the vocab (list of tuples) in the format used in the original
    hmpnn code by separating one tuple on each line
    """
    with open(filename, 'w') as f:
        for x,y in vocab:
            f.write(f"{x} {y}\n")


def load_vocab_from_txt_file(vocab_file):
    """ Reads the vocab (list of tuples) in the format used in the original 
    hmpnn code; by separating the tuple before writting a new line
    """
    with open(vocab_file) as f:
        vocab = [tuple(x.strip("\n").split()) for x in f]
    return vocab



if __name__=="__main__":
    path_to_feather_files = [
        ""
    ]
    
    pass
