from rdkit import Chem
import argparse
import pickle
import gzip
import os
import os.path as osp
import numpy as np
import torch
import tqdm
from torch.distributions.categorical import Categorical

from LambdaZero import chem
from LambdaZero.utils import get_external_dirs

from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
#import model_atom, model_block, model_fingerprint
from main_flow import make_model, Proxy

parser = argparse.ArgumentParser()

parser.add_argument("--path", default="results/_0/")
parser.add_argument("--num_gen", default=50, type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--cpu_req", default=8)

class Docker:
    def __init__(self, tmp_dir, cpu_req=2):
        self.target_norm = [-8.6, 1.10]
        self.dock = chem.DockVina_smi(tmp_dir, cpu_req=cpu_req)

    def eval(self, mol, norm=False, s=None):
        try:
            if s is None:
                s = Chem.MolToSmiles(mol.mol)
            _, r, _ = self.dock.dock(s)
        except Exception as e: # Sometimes the prediction fails
            print('exception for', s, e)
            r = 0
        if not norm:
            return r
        reward = -(r-self.target_norm[0])/self.target_norm[1]
        return reward

def main(args):
    info = pickle.load(gzip.open(f'{args.path}/info.pkl.gz'))
    params = pickle.load(gzip.open(f'{args.path}/params.pkl.gz'))
    eargs = info['args']

    device = torch.device(args.device)

    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    mdp = MolMDPExtended(bpath)
    mdp.post_init(device, eargs.repr_type)
    mdp.build_translation_table()
    mdp.floatX = torch.float if params[0].dtype == np.float32 else torch.double

    model = make_model(eargs, mdp)

    for a,b in zip(params, model.parameters()):
        b.data = torch.tensor(a)
    model.to(device)
    os.makedirs('./tmp', exist_ok=True)
    docker = Docker('./tmp', cpu_req=args.cpu_req)

    nblocks = mdp.num_blocks
    np.random.seed(123)
    torch.random.manual_seed(1234)

    gen = []
    while len(gen) < args.num_gen:
        mol = BlockMoleculeDataExtended()
        for i in range(eargs.max_blocks):
            s = mdp.mols2batch([mdp.mol2repr(mol)])
            stem_o, mol_o = model(s)
            logits = torch.cat([stem_o.flatten(), mol_o.flatten()])
            if i < eargs.min_blocks:
                logits[-1] = -20
            cat = Categorical(logits=logits)
            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                break
            else:
                act = (act % nblocks, act // nblocks)
                mol = mdp.add_block_to(mol, block_idx=act[0], stem_idx=act[1])
            if not len(mol.stems):
                break
        if mol.mol is None:
            print('skip', mol.blockidxs, mol.jbonds)
            continue
        score = docker.eval(mol)
        print(mol.smiles, score)
        gen.append([score, mol.as_dict()])
    pickle.dump(gen, gzip.open(f'{args.path}/gen_mols_{args.num_gen}.pkl.gz', 'wb'))

def main_proxy(args):
    info = pickle.load(gzip.open(f'{args.path}/info.pkl.gz'))
    params = pickle.load(gzip.open(f'{args.path}/params.pkl.gz'))
    eargs = info['args']
    print(args)
    device = torch.device(args.device)

    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    mdp = MolMDPExtended(bpath)
    mdp.post_init(device, eargs.repr_type)
    mdp.floatX = torch.double

    model = make_model(eargs, mdp)

    for a,b in zip(params, model.parameters()):
        b.data = torch.tensor(a, dtype=mdp.floatX)
    model.to(device)

    proxy = Proxy(eargs, bpath, device)


    mdp.post_init(device, eargs.repr_type)
    mdp.build_translation_table()
    nblocks = mdp.num_blocks
    import numpy as np
    np.random.seed(123)
    torch.random.manual_seed(1234)

    gen = []
    adv = tqdm.tqdm(range(args.num_gen))
    while len(gen) < args.num_gen:
        mol = BlockMoleculeDataExtended()
        for i in range(eargs.max_blocks):
            s = mdp.mols2batch([mdp.mol2repr(mol)])
            with torch.no_grad():
                stem_o, mol_o = model(s)
            logits = torch.cat([stem_o.flatten(), mol_o.flatten()])
            if i < eargs.min_blocks:
                logits[-1] = -20
            #if len(gen) < 3:
            #    print(logits.mean(), logits.min(), logits.max())
            #    print(mol_o)
            cat = Categorical(logits=logits)
            act = cat.sample().item()
            if act == logits.shape[0] - 1:
                break
            else:
                act = (act % nblocks, act // nblocks)
                mol = mdp.add_block_to(mol, block_idx=act[0], stem_idx=act[1])
            if not len(mol.stems):
                break
        if mol.mol is None:
            #print('skip', mol.blockidxs, mol.jbonds)
            continue
        score = proxy(mol)
        #print(mol.smiles, score)
        gen.append([score, mol.as_dict()])
        adv.update()
    pickle.dump(gen, gzip.open(f'{args.path}/gen_mols_proxy_{args.num_gen}.pkl.gz', 'wb'))

def gen_small_mols(args):
    gen = []
    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    mdp = MolMDPExtended(bpath)
    mdp.post_init(args.device, 'block')
    os.makedirs('./tmp', exist_ok=True)
    docker = Docker('./tmp', cpu_req=args.cpu_req)

    m = BlockMoleculeDataExtended()
    m2d = lambda m, s:{'blockidxs': m.blockidxs,
                       'dockscore': s,
                       'slices': m.slices,
                       'jbonds': m.jbonds,
                       'stems': m.stems}
    #gen.append(m2d(m, 0))
    for i in range(mdp.num_true_blocks):
        i = mdp.add_block_to(m, mdp.block_smi.index(mdp.true_block_set[i]))
        for j in range(mdp.num_blocks):
            if np.random.random() > 0.05:
                continue
            j =  mdp.add_block_to(i, j, np.random.randint(len(i.stems)))
            score = docker.eval(j, norm=False)
            print(j.blockidxs, j.smiles, score)
            gen.append(m2d(j, score))
        #gen.append(m2d(i, score))
    pickle.dump(gen, gzip.open(f'small_mols_2.05.pkl.gz', 'wb'))



if __name__ == '__main__':
    args = parser.parse_args()
    #main_proxy(args)
    #gen_small_mols(args)
    main(args)
