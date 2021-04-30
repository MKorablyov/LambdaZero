import torch
import dgl
import LambdaZero.examples.egnn.egnn as egnn
import LambdaZero.inputs as inputs
from LambdaZero.environments import MolMDP
from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Batch

torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
import os, pickle, gzip, time
import os.path as osp
try:
    from tqdm import tqdm
except:
    tqdm = lambda a, **k: a

datasets_dir, programs_dir, summaries_dir = get_external_dirs()
bpath = f'{datasets_dir}/fragdb/blocks_PDB_105.json'

def main():
    # number of random molecules we test
    N = 5_000

    # current model hyperparameters, but it could be interesting to
    # play with these values:
    batch_size = 96
    n_layers = 3
    m_dim = 128

    gnn = egnn.EGNNet(n_layers=n_layers, m_dim=m_dim,
                      backend='dgl')
    opt = torch.optim.SGD(gnn.parameters(), lr=1e-3)

    dgl_graphs, tg_graphs = get_graphs(N)
    
    print("Creating batches")
    dgl_batches, tg_batches = [], []
    for i in range(int(np.ceil(N / batch_size))):
        dgl_batches.append(dgl.batch(dgl_graphs[i*batch_size:(i+1)*batch_size]))
        tg_batches.append(Batch.from_data_list(tg_graphs[i*batch_size:(i+1)*batch_size]))


    print("Running 1 DGL epoch")
    gnn.backend='dgl'
    for i in gnn.egnn_layers:
        i.backend='dgl'

    t0 = time.time()
    for mbi in range(len(dgl_batches)):
        y = gnn(dgl_batches[mbi])
        y.sum().backward() # simulate backward pass
        opt.zero_grad()
    t1 = time.time()
    print(f"Took {t1-t0:.3f}s")



    print("Running 1 TG epoch")
    gnn.backend='torch_geometric'
    for i in gnn.egnn_layers:
        i.backend='torch_geometric'

    t0 = time.time()
    for mbi in range(len(dgl_batches)):
        y = gnn(tg_batches[mbi])
        y.sum().backward() # simulate backward pass
        opt.zero_grad()
    t1 = time.time()
    print(f"Took {t1-t0:.3f}s")
                      


def gen_mol(i):
    # Generate a random molecule
    mdp = MolMDP(bpath)
    np.random.seed(142857 + i)
    mdp.reset()
    mdp.random_walk(np.random.randint(1,8))
    smi = mdp.molecule.smiles
    return (inputs.mol_to_graph(smi, backend='dgl'), 
            inputs.mol_to_graph(smi))

def get_graphs(N):
    # Retrieve or create N random molecule graphs
    try:
        dgl_graphs, tg_graphs = pickle.load(gzip.open(
            f'/export/mila/lustre/LambdaZero/random_mols_{N}.pkl.gz'))
    except:
        print(f"Creating {N:,} random molecules...")
        # This is fairly expensive, around 0.02s/graph
        dgl_graphs = []
        tg_graphs = []
        import multiprocessing as mp
        with mp.pool.Pool(int(os.environ['NCPUS']), maxtasksperchild=1) as pool:
            it = pool.imap(gen_mol, range(N), 1)
            for i, (dgl_mol, tg_mol) in tqdm(enumerate(it), total=N):
                dgl_graphs.append(dgl_mol)
                tg_graphs.append(tg_mol)
            
            pickle.dump((dgl_graphs, tg_graphs), gzip.open(
                f'/export/mila/lustre/LambdaZero/random_mols_{N}.pkl.gz', 'wb'))
            os.system(f'chmod ugo+rw /export/mila/lustre/LambdaZero/random_mols_{N}.pkl.gz')
    return dgl_graphs, tg_graphs
  

    

if __name__ == '__main__':
    main()
