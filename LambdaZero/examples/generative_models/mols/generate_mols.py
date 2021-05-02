from rdkit import Chem
import argparse
import pickle
import gzip
import os
import os.path as osp

import torch
from torch.distributions.categorical import Categorical

from LambdaZero import chem
from LambdaZero.utils import get_external_dirs

from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
import model_atom, model_block, model_fingerprint

parser = argparse.ArgumentParser()

parser.add_argument("--path", default="results/_0/")
parser.add_argument("--num_gen", default=100)
parser.add_argument("--device", default="cuda")
parser.add_argument("--cpu_req", default=8)

class Docker:
    def __init__(self, tmp_dir, cpu_req=2):
        self.target_norm = [-8.6, 1.10]
        self.dock = chem.DockVina_smi(tmp_dir, cpu_req=cpu_req)

    def eval(self, mol):
        s = "None"
        try:
            s = Chem.MolToSmiles(mol.mol)
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        return reward

def main(args):
    info = pickle.load(gzip.open(f'{args.path}/info.pkl.gz'))
    params = pickle.load(gzip.open(f'{args.path}/params.pkl.gz'))
    eargs = info['args']

    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    mdp = MolMDPExtended(bpath)

    if eargs.repr_type == 'block_graph':
        raise ValueError('reimplement me')
        model = model_block.GraphAgent(nhid=eargs.nemb,
                                       nvec=0,
                                       num_out_per_stem=mdp.num_blocks,
                                       num_out_per_mol=1,
                                       num_conv_steps=eargs.num_conv_steps,
                                       mdp_cfg=mdp,
                                       version=eargs.model_version)
    elif eargs.repr_type == 'atom_graph':
        model = model_atom.MolAC_GCN(nhid=eargs.nemb,
                                     nvec=0,
                                     num_out_per_stem=mdp.num_blocks,
                                     num_out_per_mol=1,
                                     num_conv_steps=eargs.num_conv_steps,
                                     version=eargs.model_version)
    elif eargs.repr_type == 'morgan_fingerprint':
        raise ValueError('reimplement me')
        model = model_fingerprint.MFP_MLP(args.nemb, 3, mdp.num_blocks, 1)


    for a,b in zip(params, model.parameters()):
        b.data = torch.tensor(a)
    device = torch.device(args.device)
    model.to(device)
    os.makedirs('./tmp', exist_ok=True)
    docker = Docker('./tmp', cpu_req=args.cpu_req)

    mdp.post_init(device, eargs.repr_type)
    mdp.build_translation_table()
    nblocks = mdp.num_blocks
    import numpy as np
    np.random.seed(123)
    torch.random.manual_seed(1234)

    gen = []
    for i in range(args.num_gen):
        mol = BlockMoleculeDataExtended()
        for i in range(10):
            s = mdp.mols2batch([mdp.mol2repr(mol)])
            stem_o, mol_o = model(s)
            logits = torch.cat([stem_o.flatten(), mol_o.flatten()])
            cat = Categorical(logits=logits)
            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                break
            else:
                act = (act % nblocks, act // nblocks)
                mol = mdp.add_block_to(mol, block_idx=act[0], stem_idx=act[1])
            if not len(mol.stems):
                break
        print(mol.smiles)
        score = docker.eval(mol)
        gen.append([gen, mol.as_dict()])
    pickle.dump(gen, gzip.open(f'{args.path}/gen_mols_{args.num_gen}.pkl.gz', 'wb'))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
