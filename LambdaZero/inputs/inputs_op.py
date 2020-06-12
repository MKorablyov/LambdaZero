import os

import numpy as np
import pandas as pd
import ray
import torch
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from torch_sparse import coalesce

rdBase.DisableLog("rdApp.error")
from torch_geometric.data import InMemoryDataset, Data
import LambdaZero.chem


def onehot(arr, num_classes, dtype=np.int):
    arr = np.asarray(arr, dtype=np.int)
    assert len(arr.shape) == 1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr


def mpnn_feat(mol, ifcoord=True, panda_fmt=False):
    atomtypes = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bondtypes = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    natm = len(mol.GetAtoms())
    # featurize elements
    atmfeat = pd.DataFrame(
        index=range(natm),
        columns=[
            "type_idx",
            "atomic_number",
            "acceptor",
            "donor",
            "aromatic",
            "sp",
            "sp2",
            "sp3",
            "num_hs",
        ],
    )

    # featurize
    fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = atomtypes.get(atom.GetSymbol(), 5)
        atmfeat["type_idx"][i] = onehot([type_idx], num_classes=len(atomtypes) + 1)[0]
        atmfeat["atomic_number"][i] = atom.GetAtomicNum()
        atmfeat["aromatic"][i] = 1 if atom.GetIsAromatic() else 0
        hybridization = atom.GetHybridization()
        atmfeat["sp"][i] = 1 if hybridization == HybridizationType.SP else 0
        atmfeat["sp2"][i] = 1 if hybridization == HybridizationType.SP2 else 0
        atmfeat["sp3"][i] = 1 if hybridization == HybridizationType.SP3 else 0
        atmfeat["num_hs"][i] = atom.GetTotalNumHs(includeNeighbors=True)

    # get donors and acceptors
    atmfeat["acceptor"].values[:] = 0
    atmfeat["donor"].values[:] = 0
    feats = factory.GetFeaturesForMol(mol)
    for j in range(0, len(feats)):
        if feats[j].GetFamily() == "Donor":
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                atmfeat["donor"][k] = 1
        elif feats[j].GetFamily() == "Acceptor":
            node_list = feats[j].GetAtomIds()
            for k in node_list:
                atmfeat["acceptor"][k] = 1
    # get coord
    if ifcoord:
        coord = np.asarray(
            [mol.GetConformer(0).GetAtomPosition(j) for j in range(natm)]
        )
    else:
        coord = None
    # get bonds and bond features
    bond = np.asarray(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    )
    bondfeat = [bondtypes[bond.GetBondType()] for bond in mol.GetBonds()]
    bondfeat = onehot(bondfeat, num_classes=len(bondtypes))

    # convert atmfeat to numpy matrix
    if not panda_fmt:
        type_idx = np.stack(atmfeat["type_idx"].values, axis=0)
        atmfeat = atmfeat[
            [
                "atomic_number",
                "acceptor",
                "donor",
                "aromatic",
                "sp",
                "sp2",
                "sp3",
                "num_hs",
            ]
        ]
        atmfeat = np.concatenate([type_idx, atmfeat.to_numpy(dtype=np.int)], axis=1)
    return atmfeat, coord, bond, bondfeat


def _mol_to_graph(atmfeat, coord, bond, bondfeat, props={}):
    "convert to PyTorch geometric module"
    natm = atmfeat.shape[0]
    # transform to torch_geometric bond format; send edges both ways; sort bonds
    atmfeat = torch.tensor(atmfeat, dtype=torch.float32)
    edge_index = torch.tensor(
        np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64
    )
    edge_attr = torch.tensor(
        np.concatenate([bondfeat, bondfeat], axis=0), dtype=torch.float32
    )
    edge_index, edge_attr = coalesce(edge_index, edge_attr, natm, natm)
    # make torch data
    if coord is not None:
        coord = torch.tensor(coord, dtype=torch.float32)
        data = Data(
            x=atmfeat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props
        )
    else:
        data = Data(x=atmfeat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data


def mol_to_graph(smiles, props={}, num_conf=1, noh=True, feat="mpnn"):
    "mol to graph convertor"
    mol, _, _ = LambdaZero.chem.build_mol(smiles, num_conf=num_conf, noh=noh)
    if feat == "mpnn":
        atmfeat, coord, bond, bondfeat = mpnn_feat(mol)
    else:
        raise NotImplementedError(feat)
    graph = _mol_to_graph(atmfeat, coord, bond, bondfeat, props)
    return graph


@ray.remote
def _brutal_dock_proc(smi, props, pre_filter, pre_transform):
    try:
        graph = mol_to_graph(smi, props)
    except Exception as e:
        return None
    if pre_filter is not None and not pre_filter(graph):
        return None
    if pre_transform is not None:
        graph = pre_transform(graph)
    return graph


class BrutalDock(InMemoryDataset):
    # own internal dataset
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        props=["gridscore"],
        file_names=["ampc_100k"],
        chunksize=550,
    ):
        self._props = props
        self.file_names = file_names
        self._chunksize = chunksize
        super(BrutalDock, self).__init__(root, transform, pre_transform, pre_filter)

        #  todo: store a list, but have a custom collate function on batch making
        graphs = []
        for processed_path in self.processed_paths:
            self.data, self.slices = torch.load(processed_path)
            graphs += [self.get(i) for i in range(len(self))]
        if len(graphs) > 0:
            self.data, self.slices = self.collate(graphs)

    @property
    def raw_file_names(self):
        return [file_name + ".feather" for file_name in self.file_names]

    @property
    def processed_file_names(self):
        return [file_name + ".pt" for file_name in self.file_names]

    def download(self):
        pass

    def process(self):
        print("processing", self.raw_paths)
        for i in range(len(self.raw_file_names)):
            if not os.path.exists(self.processed_file_names[i]):
                docked_index = pd.read_feather(self.raw_paths[i])
                smis = docked_index["smiles"].tolist()
                props = {pr: docked_index[pr].tolist() for pr in self._props}
                tasks = [
                    _brutal_dock_proc.remote(
                        smis[j],
                        {pr: props[pr][j] for pr in props},
                        self.pre_filter,
                        self.pre_transform,
                    )
                    for j in range(len(smis))
                ]
                graphs = ray.get(tasks)
                graphs = [g for g in graphs if g is not None]
                # save to the disk
                torch.save(self.collate(graphs), self.processed_paths[i])
