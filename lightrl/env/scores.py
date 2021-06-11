import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
import torch
from LambdaZero.models import ChempropWrapper_v1
from chemprop.features import BatchMolGraph, MolGraph
from torch_geometric.data import Data, Batch
from lightrl.env.vec_env import fast_from_data_list
from torch_geometric.utils import degree


class QEDEstimator:
    def __init__(self):
        pass

    def eval(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        qed = QED.qed(mol)
        return qed

    def __call__(self, smiles):
        ret = np.array([self.eval(x) for x in smiles])
        return ret


class ParallelSynth(torch.nn.Module):
    def __init__(self, synth_config=None, use_cuda=True):
        super(ParallelSynth, self).__init__()

        if synth_config is None:
            from lightrl.env.block_mol_v3 import synth_config

        self._synth_net = ChempropWrapper_v1(synth_config, use_cuda=use_cuda)

        self._synth_net.model.eval()

    def __call__(self, mols):
        mols = BatchMolGraph([MolGraph(x) for x in mols])
        return self.eval_batch(mols)

    def eval_batch(self, mols):
        model = self._synth_net.model
        features = self._synth_net.predict_config["features_generator"]

        with torch.no_grad():
            preds = model(mols, features)

        preds = preds.data.cpu().numpy()
        if self._synth_net.scaler is not None:
            preds = self._synth_net.scaler.inverse_transform(preds)

        return preds.reshape(-1)


class ProxyDockNetRun(torch.nn.Module):
    def __init__(self, model, transform):
        super(ProxyDockNetRun, self).__init__()

        self._model = model
        self._model.eval()
        self._transform = transform
        self.edge_weidht = False

    def forward(self, data, new_batch_graph=True, norm=True):
        if isinstance(data, Data):
            # Create batch (this will create new Batch so no problem overwriting the edge_index
            device = data.x.device
            data = fast_from_data_list([data])
            data = data.to(device)
        elif new_batch_graph:
            # TODO copy to new batch
            raise NotImplementedError

        if self._transform is not None:
            data = self._transform(data)

        if self.edge_weidht:

            row, col = data.edge_index
            data.edge_weight = 1. / degree(col, data.num_nodes)[col]  # Norm by in-degree.

        with torch.no_grad():
            dockok, _ = self._model(data)

        docks = dockok.detach().cpu().numpy()
        if not norm:
            docks = self._model._norm.itfm(docks)

        return docks


def load_docknet(docknetpath, device):
    from lightrl.reg_models import get_reg_model
    from train_reg_light import ListGraphDBDataset
    import pandas as pd

    checkpoint = torch.load(docknetpath)
    cfg = checkpoint["cfg"]

    model = get_reg_model(cfg.model)
    model.to(device)
    model.load_state_dict(checkpoint["state_dict"])

    # Used to calculate graph
    db = ListGraphDBDataset(pd.DataFrame(), cfg.dataset)
    model._norm = db._norm

    proxy_net = ProxyDockNetRun(model, db.transform)
    return proxy_net
