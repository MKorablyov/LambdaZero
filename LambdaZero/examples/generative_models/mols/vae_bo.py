from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, qUpperConfidenceBound
from botorch.sampling import IIDNormalSampler
from botorch.sampling.samplers import SobolQMCNormalSampler
import torch

from tqdm import tqdm
import time, gzip, pickle
import os.path as osp
import numpy as np

from LambdaZero.utils import get_external_dirs
from mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from main_flow import Proxy, make_model

def main():
    datasets_dir, programs_dir, summaries_dir = get_external_dirs()
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")

    info = pickle.load(gzip.open('results/_0/info.pkl.gz', 'rb'))
    args = info['args']
    device = torch.device('cpu')

    mdp = MolMDPExtended(bpath)
    mdp.post_init(device, args.repr_type, include_nblocks=args.include_nblocks)
    mdp.build_translation_table()
    mdp.floatX = torch.double

    proxy = Proxy(args, bpath, device)

    model = make_model(args, mdp, out_per_mol=args.nembmol*2+1, nvec=args.nembmol)
    model.to(args.floatX)
    model.to(device)

    min_blocks = args.min_blocks
    max_blocks = args.max_blocks

    def decode_x(x):
        return [decode_one_x(i) for i in x]

    def decode_one_x(x):
        m = BlockMoleculeDataExtended()
        for t in range(max_blocks): 
            s = mdp.mols2batch([mdp.mol2repr(m)])
            s_o, m_o = model(s, vec_data=x[None, :])
            m_o = m_o[:,0]
            if t < min_blocks:
                m_o = m_o * 0 - 1000 # prevent assigning prob to stop
                                     # when we can't stop
            ##
            logits = torch.cat([m_o.reshape(-1), s_o.reshape(-1)])
            cat = torch.distributions.Categorical(
                logits=logits)
            action = cat.sample().item()
            if t >= min_blocks and action == 0:
                break
            else:
                action = max(0, action-1)
                action = (action % mdp.num_blocks, action // mdp.num_blocks)
                m_old = m
                m = mdp.add_block_to(m, *action)
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    break
        return m

    params = pickle.load(gzip.open('results/_0/params.pkl.gz', 'rb'))
    for a,b in zip(model.parameters(), params):
        a.data = torch.tensor(b)

    state_dict = None

    mbsize = 32

    x = torch.randn(mbsize, args.nembmol)
    y = torch.tensor([proxy(i) / 10 for i in decode_x(x)])
    
    
    for i in range(100):

        gp = SingleTaskGP(train_X=x, train_Y=y.unsqueeze(1))
        gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

        if state_dict is not None:
            gp.load_state_dict(state_dict)
        mll = ExactMarginalLogLikelihood(likelihood=gp.likelihood, model=gp)
        mll = mll.to(x)
        
        t0 = time.time()
        opt = torch.optim.Adam(gp.parameters(), 5e-2)
        gp.train()
        pg = tqdm(range(1000))
        ema_loss = None
        for j in pg:
            opt.zero_grad()
            out = gp(x)
            loss = -mll(out, gp.train_targets)
            loss.backward()
            ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
            pg.set_description(f'{ema_loss:.3f} {loss.item():.5f}')
            opt.step()
        t1 = time.time()
        state_dict = gp.state_dict()
        gp.eval()

        sampler = IIDNormalSampler(num_samples=100, resample=True)
        #sampler = SobolQMCNormalSampler(num_samples=2048*8, seed=42)
        #qEI = qExpectedImprovement(gp, best_f=y.max(), sampler=sampler)
        qEI = qUpperConfidenceBound(gp, 0.1, sampler)
        X = torch.randn(mbsize*4, 4, args.nembmol) # was mbsize//4
        X.requires_grad_()
        X0 = (X + 0).data.clone()
        xopt = torch.optim.Adam([X], 1e-1) # was 1e-3
        ema_loss = None
        pg = tqdm(range(2500))
        for j in pg:
            xopt.zero_grad()
            loss = -qEI(X).sum()
            loss.backward()
            ema_loss = loss.item() if ema_loss is None else 0.99 * ema_loss + 0.01 * loss.item()
            pg.set_description(f'{ema_loss:.3f} {loss.item():.5f} {X.grad.data.pow(2).sum().sqrt()}')
            xopt.step()
        print((X-X0).pow(2).sum().sqrt())
        t2 = time.time()
        X = X.reshape((-1, args.nembmol))
        x = torch.cat([x, X], 0)
        y = torch.cat([y, torch.tensor([proxy(i) / 10 for i in decode_x(X)])])
        t3 = time.time()
        sortedy = sorted(y)
        print(i, f'{t1-t0:.1f} {t2-t1:.1f} {t3-t2:.1f} -- {y.max():.2f} {np.mean(sortedy[-10:]):.2f}')

if __name__ == '__main__':
    main()
        
