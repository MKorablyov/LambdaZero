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
    for smi in list_of_smiles:
        hmol = MolGraph(smi)
        for _, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for _, s in attr['inter_label']:
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
    This model class adapts the hierarchical mpnn model from hmpnn.
    The Forward method assumes that the batch of molecules contains
    a field "smiles" to extract features, as for chemprop.

    there is a fundamental problem with the way this model works.
    a vocabulary constructed from the dataset needs to be passed to the model.
    In effect, this corresponds to a dependency on the dataset, which is not
    allowed by how an experiment usually work.

    To circumvent the problem, the name of the dataset, by default 
    "D4MoleculesDataset", is passed to the model. A corresponding vocab file 
    `D4MoleculesDataset_vocab.txt` is loaded. That file should be considered
    part of the model. To extend thi model to other datasets, new vocab.txt 
    file will be required.
    """

    def __init__(self, dataset_name="D4MoleculesDataset", rnn_type='LSTM', embed_size=270, hidden_size=270, depthT=20, depthG=20, dropout=0.0):
        super().__init__()
        
        model_dir = Path(__file__).resolve().parent
        vocab_file_path = model_dir.joinpath(f"{dataset_name}_vocab.txt")

        if not vocab_file_path.exists():
            raise NotImplementedError("using HierarchMPNN with a new dataset requires to make the corresponding _vocab.txt first")
        
        vocab = load_vocab_from_txt_file(str(vocab_file_path))
        self.vocab = PairVocab(vocab, cuda=False)
        self.avocab = common_atom_vocab
        self.hgraph_encoder = HierMPNEncoder(self.vocab, self.avocab, rnn_type, embed_size, hidden_size, depthT, depthG, dropout)

        # self.set2set_1 = Set2Set(gcn_size, processing_steps=3)
        # self.set2set_1 = Set2Set(gcn_size, processing_steps=3)
        # self.set2set_1 = Set2Set(gcn_size, processing_steps=3)
        
        # self.fully_connected = nn.Sequential(
        #     nn.Linear(2*gcn_size, linear_hidden),
        #     nn.ReLU(),
        #     nn.Linear(linear_hidden, out_size)
        # )

    def forward(self, batch: Batch):
        smiles_list = get_list_of_smiles_from_batch(batch)

        # wasted (tree_batchG, graph_batchG), all_order computation from MolGraph.tensorize
        batchG_tuple , tensors_tuple, all_order = MolGraph.tensorize(smiles_list, self.vocab, self.avocab)
        tree_tensors, graph_tensors = tensors_tuple
        
        # wasted inter_vecs computation from HierMPNEncoder.forward
        root_vecs, node_vecs, inter_vecs, atom_vecs = self.hgraph_encoder(tree_tensors, graph_tensors)
        
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


if __name__=="__main__": ## for debugging
    """
    making the original "_vocab.txt" file and example batch
    """
    from LambdaZero.datasets.brutal_dock.datasets import D4MoleculesDataset

    dataset_name="D4MoleculesDataset"
    root_dir = "/Users/Simon/codes/LambdaZero/LambdaZero/datasets/brutal_dock/experiments/data/"
    raw_dir = "/Users/Simon/codes/LambdaZero/Datasets/brutal_dock/d4/raw/"
    raw_files = ['dock_blocks105_walk40_clust.feather', 'dock_blocks105_walk40_2_clust.feather']
    raw_paths = [raw_dir + name for name in raw_files]
    
    model_dir = Path(__file__).resolve().parent
    vocab_file_path = model_dir.joinpath(f"{dataset_name}_vocab.txt")

    if not vocab_file_path.exists():
        smiles_list = feather_to_smiles_list(raw_paths)
        vocab = build_vocab_from_simles_list(smiles_list)
        vocab_to_txt_file(vocab, str(vocab_file_path))

    dataset = D4MoleculesDataset(root_dir, raw_dir)
    
    parameters = {
        'dataset_name': 'D4MoleculesDataset',
        'rnn_type': "LSTM",
        'embed_size': 12,
        'hidden_size': 12,
        'depthT': 20,
        'depthG': 20,
        'dropout': 0.0,
    }

    batch = dataset[:16]
    net = HierarchMPNN(**parameters)
    out = net(batch)
    print(out)