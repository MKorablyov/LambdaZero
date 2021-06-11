from lightrl import env
import gym
from lightrl.env.vec_env import get_envs, pre_process_obss
import torch
from lightrl.env.vec_env import fast_from_data_list
from argparse import Namespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set


from lightrl.reg_models import mpn_model
from lightrl.reg_models import model_reg_atom
from lightrl.reg_models import model_egnn
from lightrl.reg_models.reg_with_feat_ext import RegBase
from lightrl.reg_models.actor_base_model import ActorWrapper


MODELS = {
    "MPNNet": mpn_model.MPNNet,
    "MPNNet_v2": model_reg_atom.MPNNet_v2,
    "EGNNetRLBO": model_egnn.EGNNetRLBO,
}

import glob
from importlib import import_module

python_paths = glob.glob("lightrl/reg_models/geometric_examples/*.py")
var_name = "network"

for python_path in python_paths:
    module_path = python_path.replace("/", ".").replace(".py", "")

    mod = import_module(module_path)
    if hasattr(mod, var_name):

        if isinstance(getattr(mod, var_name), list):
            _models = getattr(mod, var_name)
        else:
            _models = [getattr(mod, var_name)]

        for network_name, network in _models:
            assert network_name not in MODELS, f"Network name already exists {network_name} - " \
                                               f"{MODELS[network_name]} was before {network}"
            MODELS[network_name] = network

# ======================
# Get demo obs batch

env = gym.make("BlockMolEnvGraph-v1")
obs = env.reset()

data = fast_from_data_list([obs["mol_graph"]])
data.smiles = [env.molMDP.molecule.smiles]
# ======================


# ======================
m_name = "SGC"

cfg = Namespace(name=m_name, num_feat=14, dim=128, dropout=0., num_out=2)



data = data.to(torch.device("cuda"))

from lightrl.reg_models import get_actor_model
model = get_actor_model(cfg)
model.cuda()
out = model(data)
exit()

model = MODELS[m_name](cfg)
model.cuda()

out = model(data)
print("\n", m_name)
print("feat extr", out.shape)

if MODELS[m_name].atom_feature_extractor:
    reg_model = RegBase(cfg, MODELS[m_name])
    reg_model.cuda()
    out, per_atom_out = reg_model(data)
    print(out.size(), per_atom_out.size(), "||", reg_model.per_atom_out_size)
    print("reg M", out.shape)

exit()
# ======================
print(MODELS.keys())

data = data.to(torch.device("cuda"))
def test():
    from copy import deepcopy
    cfg = Namespace(num_feat=14, dim=128, dropout=0, num_out=2)

    for m_name in MODELS.keys():
        print("\n", m_name)

        model = MODELS[m_name](cfg)
        model.cuda()
        out = model(deepcopy(data))
        print("feat extr", out.shape)

        if MODELS[m_name].atom_feature_extractor:
            reg_model = RegBase(cfg, MODELS[m_name])
            reg_model.cuda()
            out = reg_model(data)
            print("reg M", out.shape)

# test()