from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Data, InMemoryDataset
from pubchempy import Compound
import urllib.request
import ssl
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm


class NewDrugComb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, n_laplace_feat=256, **kwargs):

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.n_laplace_feat = n_laplace_feat

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + '/NewDrugComb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        # Log transform concentrations
        self.data.ddi_edge_attr[:, 1] = torch.log(self.data.ddi_edge_attr[:, 1] + 1e-6)
        self.data.ddi_edge_attr[:, 2] = torch.log(self.data.ddi_edge_attr[:, 2] + 1e-6)

    @property
    def raw_file_names(self):
        return [
            'drugcomb_data_v1.4.csv',
            'summary_table_v1.4.csv'
        ]

    @property
    def processed_file_names(self):
        return ['drugcomb_data.pt']

    def download(self):
        urls = [
            'https://drugcomb.fimm.fi/download/drugcomb_data_v1.4.csv',
            'https://drugcomb.fimm.fi/download/summary_table_v1.4.csv'
        ]

        chunk_size = 16 * 1024
        for url in urls:
            file_name = url.split('/')[-1]
            gcontext = ssl.SSLContext()
            with urllib.request.urlopen(url, context=gcontext) as f:
                with open(os.path.join(self.raw_dir, file_name), "w") as file:
                    while True:
                        file_chunk = f.read(chunk_size).decode('utf-8')
                        if not file_chunk:
                            break
                        file.write(file_chunk)

    def process(self):

        # Load dataframes
        self._drugcomb_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]), low_memory=False)
        self.cid_to_smiles_dict = self._get_smiles()
        self.nodes, cid_to_idx_dict = self._get_nodes()

        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes = self._get_ddi_edges(cid_to_idx_dict)

        data = Data(x_drugs=torch.tensor(self.nodes.to_numpy(), dtype=torch.float))

        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.float)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.float)

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_smiles(self):
        unique_cid = set(self._drugcomb_data['drug_row_cid'].unique()).\
            union(self._drugcomb_data['drug_col_cid'].unique())

        unique_cid.remove(0)
        unique_cid = list(unique_cid)

        dict_path = os.path.join(self.raw_dir, "cid_to_smiles_dict.npy")
        if os.path.exists(dict_path):
            cid_to_smiles_dict = np.load(dict_path, allow_pickle=True).item()
        else:
            print('recovering smiles, only happens the first time...')
            cid_to_smiles_dict = {}
            for cid in tqdm(unique_cid):
                try:
                    cid_to_smiles_dict[int(cid)] = Compound.from_cid(int(cid)).isomeric_smiles
                except:
                    print("error with cid", cid)
                    pass
            np.save(dict_path, cid_to_smiles_dict)

        return cid_to_smiles_dict

    def _get_fingerprint(self, smile, radius, n_bits):
        if smile == 'none':
            return np.array([-1] * n_bits)
        try:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
        except Exception as ex:
            return np.array([-1] * n_bits)

    def _get_nodes(self):

        self._drugcomb_data['drug_row_smiles'] = self._drugcomb_data['drug_row_cid'].apply(
            lambda cid: self.cid_to_smiles_dict[cid] if cid in self.cid_to_smiles_dict.keys() else -1)

        self._drugcomb_data['drug_col_smiles'] = self._drugcomb_data['drug_col_cid'].apply(
            lambda cid: self.cid_to_smiles_dict[cid] if cid in self.cid_to_smiles_dict.keys() else -1)

        # Computing fingerprints
        cid_to_fp_dict = {cid: self._get_fingerprint(self.cid_to_smiles_dict[cid], self.fp_radius, self.fp_bits)
                          for cid in self.cid_to_smiles_dict.keys()}

        nodes = pd.DataFrame.from_dict(cid_to_fp_dict, orient='index')

        cid_to_idx_dict = {nodes.index[i]: i for i in range(len(nodes))}
        nodes.reset_index(drop=True, inplace=True)

        return nodes, cid_to_idx_dict

    def _get_ddi_edges(self, cid_to_idx_dict):
        print('Processing drug drug interaction edges..')

        # Drop rows that do not have second drug
        self._drugcomb_data = self._drugcomb_data.dropna(axis=0, subset=['drug_col'])

        cell_line_dummies = pd.get_dummies(self._drugcomb_data['cell_line_name'])

        # Add one hot encoding of cell lines
        self._drugcomb_data = pd.concat((self._drugcomb_data, cell_line_dummies),
                                        axis=1)

        self._drugcomb_data['drug_row_idx'] = self._drugcomb_data['drug_row_cid'].\
            apply(lambda cid: cid_to_idx_dict[cid])

        self._drugcomb_data['drug_col_idx'] = self._drugcomb_data['drug_col_cid'].\
            apply(lambda cid: cid_to_idx_dict[cid])

        is_pairs_of_diff_drugs = self._drugcomb_data['drug_row_idx'] != self._drugcomb_data['drug_col_idx']
        self._drugcomb_data = self._drugcomb_data[is_pairs_of_diff_drugs]

        ddi_edge_idx = self._drugcomb_data[['drug_row_idx', 'drug_col_idx']].to_numpy().T
        ddi_edge_classes = self._drugcomb_data[cell_line_dummies.columns].to_numpy()
        ddi_edge_attr = self._drugcomb_data[['conc_r', 'conc_c', 'inhibition']].to_numpy()

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes


if __name__ == '__main__':

    dataset = NewDrugComb()

