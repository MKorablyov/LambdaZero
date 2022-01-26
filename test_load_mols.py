import torch
import os.path as osp
import numpy as np

from LambdaZero.examples.generative_models.mols.mol_mdp_ext import MolMDPExtended, BlockMoleculeDataExtended
from LambdaZero.utils import get_external_dirs


datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def get_mdp(
        bpath: str = "fragdb/fix_131.json", repr_type: str = "block_graph",
        mdp_overwrite: dict = None, device=torch.device("cpu")
) -> MolMDPExtended:
    mdp_init_cfg = {
        "bpath": bpath,
        "device": device,
        "repr_type": repr_type,
        "include_nblocks": False,
        "floatX": "float64",
    }
    if mdp_overwrite is not None:
        mdp_init_cfg.update(mdp_overwrite)

    bpath = osp.join(datasets_dir, mdp_init_cfg["bpath"])
    mdp = MolMDPExtended(bpath)  # Used for generating representation
    mdp.post_init(**mdp_init_cfg)
    mdp.build_translation_table()
    return mdp


if __name__ == "__main__":
    data = np.load("data/mols_gen_random_proxy_ProxyQED_seed_113_cnt_300000.npy", allow_pickle=True).item()
    mdp = get_mdp()
    mol = mdp.load(data[0])
