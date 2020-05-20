import torch
import torch.nn as nn
from torch_geometric.data import Batch
from hgraph import HierMPNEncoder, MolGraph, common_atom_vocab
from LambdaZero.datasets.brutal_dock.models.model_base import ModelBase, get_list_of_smiles_from_batch


def build_vocab_from_simles_list(list_of_all_smiles):
    """
    returns a set containing (smiles,  was adapted from hgraph `get_vocab.py` script
    """
    vocab = set()
    for s in list_of_all_smiles:
        hmol = MolGraph(s)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab


if __name__=="__main__":
    """
    for debugging purposes
    """

    smiles_list = ['CC(C)=CC(C)(C)O',
        'CC(C)=CC(=O)NC(C#N)P(=O)(O)O',
        'O=[SH](=O)S(=O)(=O)O',
        'CC(C)=CN1CCN(P(=O)(O)O)CC1',
        'CC(C)(O)C(F)(F)F',
        'c1ccc2cc(N3CCOCC3)ccc2c1',
        'CC(C)(O)Br',
        'CC(=O)N[SH](=O)=O',
        'CC(C)=CC1CC(C(NC(=O)C(C)O)NC(=O)S(=O)(=O)O)N(c2ccc3ccccc3c2)C1C(C)C',
        'C1=C(c2ccc[nH]2)CCCC1',
        'O=C(NCF)C1=CCCCC1',
        'CC(=Cc1cc[nH]c1)CCCl',
        'CC(=O)NC(=O)NC1=CN(I)C=CC1',
        'Nn1cnc2c([PH](=O)[O-])ncnc21',
        'N=C[n+]1cccc(-[n+]2cccc(C(NC(=O)c3csc(N4C=CCC(S)=C4)n3)[NH+]3CCOCC3)c2)c1',
        'O=C(NC([NH+]1CCOCC1)[PH](=O)O)n1ccc(S(=O)(=O)O)nc1=O',
        'CC(O)c1cn(C(C)O)c(=O)[nH]c1=O',
        'CC(=O)NCc1ccn(C2CNc3nc(-c4ccc[nH]4)[nH]c(=O)c3N2)c(=O)n1',
        'C[SH+]c1cc[nH]c1',
        'O=c1[nH]cc(-n2cnc3c(C4C(C5CNc6nc(C(F)(F)F)[nH]c(=O)c6N5)CCOC4[n+]4ccccc4)ncnc32)c(=O)[nH]1',
    ]
    vocab = build_vocab_from_simles_list(smiles_list)
    print(vocab)
