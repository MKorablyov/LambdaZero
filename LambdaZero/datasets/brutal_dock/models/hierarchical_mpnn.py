import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from hgraph import HierMPNEncoder, MolGraph, PairVocab, common_atom_vocab
from LambdaZero.datasets.brutal_dock.models.model_base import ModelBase, get_list_of_smiles_from_batch


def build_vocab_from_simles_list(list_of_smiles):
    """
    returns a list of (smiles, smiles) tuples. This function was adapted from
    hgraph's `get_vocab.py` script, that must run before using hgraph
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


class HierarchMPNN(ModelBase):
    """
    This model class adapts the hierarchical mpnn model to LambdaZero.
    The Forward method assumes that the batch of molecules contains
    a field "smiles" to extract features, as we did for chemprop.
    """

    def __init__(self, vocab_file, rnn_type='LSTM', embed_size=270, hidden_size=270, depthT=20, depthG=20, dropout=0.0):
        super().__init__()
        file_path = Path(vocab_file)

        if file_path.suffix == ".feather":
            txt_file = Path(str(file_path).replace(".feather","_vocab.txt"))
            if not txt_file.exists():
                smiles_list = feather_to_smiles_list(str(file_path))
                vocab = build_vocab_from_simles_list(smiles_list)
                vocab_to_txt_file(vocab, str(txt_file))
            file_path = txt_file
        vocab = load_vocab_from_txt_file(str(file_path))
        vocab = PairVocab(vocab)
        avocab = common_atom_vocab
        self.hgraph_encoder = HierMPNEncoder(vocab, avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)

    def forward(self, batch: Batch):
        smiles_list = get_list_of_smiles_from_batch(batch)

        # wasted (tree_batchG, graph_batchG), all_order computation from MolGraph.tensorize
        batchG_tuple , tensors_tuple, all_order = MolGraph.tensorize(smiles_list, self.vocab, self.avocab)
        tree_tensors, graph_tensors = tensors_tuple
        
        # wasted inter_vecs computation from HierMPNEncoder.forward
        root_vecs, node_vecs, inter_vecs, atom_vecs = self.encoder((tree_tensors, graph_tensors))
        
        # TODO:
        # root_vecs = aggregate(root_vecs)
        # node_vecs = aggregate(node_vecs)
        # atom_vecs = aggregate(atom_vecs)
        # r = self.mlp1(root_vecs)
        # n = self.mlp2(node_vecs)
        # a = self.mlp3(atom_vecs)
        # prediction = self.mlp_out(concat(r, n, a))
        # return prediction

        return 0


if __name__=="__main__":
    path_to_feather_files = [
        ""
    ]
    
    pass
