import os

import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd

from rdkit import Chem

import multiprocessing
import tempfile

root = "/home/lsky/datasets/zinc20"


def _parse_sdf(path):
    mol = Chem.SDMolSupplier(path)[0]
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    atoms = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=np.float32)
    bonds = np.asarray([[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()], dtype=np.int64)
    bonds = np.vstack((bonds, bonds[:, ::-1])).T
    return pos, atoms, bonds


def _parse_pdbqt(path):
    with open(path) as f:
        data = [line for line in f if line.startswith("ATOM")]
    # number of entries in line is unstable, but counting from the end works
    pos = np.array([line.split()[-7:-4] for line in data], dtype=np.float32)
    return pos


def _parse_docked_pdb(path):
    with open(path) as f:
        data = f.readlines()

    indices = [idx for idx, line in enumerate(data) if line.startswith("MODEL") or line.startswith("ENDMDL")]
    indices = np.array(indices, dtype=np.int64).reshape(-1, 2)

    energies = []
    pos = []
    for start, end in indices:
        data_sub = data[start:end]
        energies.append(float(data_sub[1].split()[3]))
        data_sub_atoms = [line for line in data_sub if line.startswith("ATOM")]
        pos.append([line.split()[-7:-4] for line in data_sub_atoms])

    energies = np.array(energies, dtype=np.float32)
    pos = np.array(pos, dtype=np.float32)
    return pos, energies


def _construct_graph(int_zinc_id, smiles, dir):
    sdf_path = os.path.join(dir, "sdf", f"{int_zinc_id}.sdf")
    pdbqt_path = os.path.join(dir, "pdbqt", f"{int_zinc_id}.pdbqt")
    pdb_path = os.path.join(dir, "docked", f"{int_zinc_id}.pdb")

    sdf_pos, sdf_atoms, sdf_bonds = _parse_sdf(sdf_path)
    pdbqt_pos = _parse_pdbqt(pdbqt_path)
    pdb_pos, pdb_energies = _parse_docked_pdb(pdb_path)

    # find reordering
    order = np.argmin(np.abs(sdf_pos.reshape((1, -1, 3)) - pdbqt_pos.reshape((-1, 1, 3))).sum(axis=2), axis=0)

    pdbqt_pos_reordered = pdbqt_pos[order]
    pdb_pos_reordered = pdb_pos[:, order]

    if not np.all(np.isclose(sdf_pos, pdbqt_pos_reordered, atol=1e-3)):
        # if reordering failed return just id, so that we know which entry failed
        graph = Data(int_zinc_id=int_zinc_id)
    else:
        z = torch.from_numpy(sdf_atoms)
        edge_index = torch.from_numpy(sdf_bonds)
        pos_free = torch.from_numpy(sdf_pos)
        pos_docked = torch.from_numpy(pdb_pos_reordered)
        dockscores = torch.from_numpy(pdb_energies)
        graph = Data(int_zinc_id=int_zinc_id, smiles=smiles, z=z, edge_index=edge_index, pos_free=pos_free, pos_docked=pos_docked, dockscores=dockscores)
    return graph


def main(batch_idx):
    try:
        records = pd.read_csv(os.path.join(root, "score_subsets", f"subset_{batch_idx}.csv"))
        records = records[records.status == "ok"][['int_zinc_id', 'smiles']]

        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = os.path.join(root, 'sdf_subsets', f'sdf_{batch_idx}.tar.gz')
            pdbqt_path = os.path.join(root, 'pdbqt_subsets', f'pdbqt_{batch_idx}.tar.gz')
            docked_path = os.path.join(root, 'docked_subsets', f'docked_{batch_idx}.tar.gz')

            os.system(f"tar -xzf {sdf_path} -C {tmpdir}")
            os.system(f"tar -xzf {pdbqt_path} -C {tmpdir}")
            os.system(f"tar -xzf {docked_path} -C {tmpdir}")

            data = [_construct_graph(int_zinc_id, smiles, tmpdir) for int_zinc_id, smiles in zip(records.int_zinc_id, records.smiles)]

        torch.save(data, os.path.join(root, "graphs", f"{batch_idx}.pth"))
        print(f"{batch_idx} success!")
    except Exception as e:
        print(f"{batch_idx} failed!")
        print(e)


if __name__ == "__main__":
    pool = multiprocessing.Pool()
    pool.map(main, range(1000))
