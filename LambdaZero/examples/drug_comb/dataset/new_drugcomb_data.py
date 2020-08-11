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
        self.data.ddi_edge_attr[:, 0] = torch.log(self.data.ddi_edge_attr[:, 0] + 1e-6)
        self.data.ddi_edge_attr[:, 1] = torch.log(self.data.ddi_edge_attr[:, 1] + 1e-6)

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

        self.name_to_cid = self._get_name_to_cid()

        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, cell_line_name_to_name_idx = \
            self._get_ddi_edges(cid_to_idx_dict)

        data = Data(x_drugs=torch.tensor(self.nodes.to_numpy(), dtype=torch.float))

        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.float)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.float)
        data.cell_line_name_to_name_idx = cell_line_name_to_name_idx

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

        dict_path = os.path.join(self.raw_dir, "new_cid_to_smiles_dict.npy")
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

    def _get_name_to_cid(self):
        first = self._drugcomb_data[['drug_row', 'drug_row_cid']].rename(columns={'drug_row': 'name', 'drug_row_cid': 'cid'})
        scnd = self._drugcomb_data[['drug_col', 'drug_col_cid']].rename(columns={'drug_col': 'name', 'drug_col_cid': 'cid'})

        uniques = first.append(scnd).dropna().drop_duplicates().values.tolist()

        return {tpl[0]: tpl[1] for tpl in uniques}

    def _get_ddi_edges(self, cid_to_idx_dict):
        print('Processing drug drug interaction edges..')

        # Drop rows that do not have second drug
        self._summary_data = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]), low_memory=False)
        self._summary_data = self._summary_data.dropna(axis=0, subset=['drug_col'])

        cell_line_dummies = pd.get_dummies(self._summary_data['cell_line_name'])

        # Add one hot encoding of cell lines
        self._summary_data = pd.concat((self._summary_data, cell_line_dummies),
                                        axis=1)

        self._summary_data['drug_row_idx'] = self._summary_data['drug_row'].\
            apply(lambda drug: cid_to_idx_dict[self.name_to_cid[drug]]
                  if drug in self.name_to_cid and self.name_to_cid[drug] in cid_to_idx_dict
                  else -1)

        self._summary_data['drug_col_idx'] = self._summary_data['drug_col'].\
            apply(lambda drug: cid_to_idx_dict[self.name_to_cid[drug]]
                  if drug in self.name_to_cid and self.name_to_cid[drug] in cid_to_idx_dict
                  else -1)

        self._summary_data = self._summary_data[self._summary_data['drug_row_idx'] != -1]
        self._summary_data = self._summary_data[self._summary_data['drug_col_idx'] != -1]

        is_pairs_of_diff_drugs = self._summary_data['drug_row_idx'] != self._summary_data['drug_col_idx']
        self._summary_data = self._summary_data[is_pairs_of_diff_drugs]

        ddi_edge_idx = self._summary_data[['drug_row_idx', 'drug_col_idx']].to_numpy().T
        ddi_edge_classes = self._summary_data[cell_line_dummies.columns].to_numpy()
        ddi_edge_attr = self._summary_data[['ic50_row', 'ic50_col', 'css']].to_numpy()

        cell_line_name_to_name_idx = {
            cell_line: idx for idx, cell_line in enumerate(cell_line_dummies.columns)
        }

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes, cell_line_name_to_name_idx

    def to_cell_line(self, cell_line_name):
        match_idxs = torch.full((self.data.ddi_edge_idx.shape[1],), False, dtype=torch.bool)
        try:
            cell_line_idx = self.data.cell_line_name_to_name_idx[0][cell_line_name]
            match_idxs = self.data.ddi_edge_classes[:, cell_line_idx] == 1

        except KeyError:
            pass

        self.data.ddi_edge_idx = self.data.ddi_edge_idx[:, match_idxs]
        self.data.ddi_edge_classes = self.data.ddi_edge_classes[match_idxs]
        self.data.ddi_edge_attr = self.data.ddi_edge_attr[match_idxs]

        return self

    def __len__(self):
        return self.data.ddi_edge_idx.shape[1]

    def __getitem__(self, idx):
        drug_row_idx = self.data.ddi_edge_idx[0, idx]
        drug_col_idx = self.data.ddi_edge_idx[1, idx]

        fp_row = self.data.x_drugs[drug_row_idx]
        fp_col = self.data.x_drugs[drug_col_idx]

        conc_row = self.data.ddi_edge_attr[idx, 0].reshape(-1, 1)
        conc_col = self.data.ddi_edge_attr[idx, 1].reshape(-1, 1)

        drug_x = torch.cat((fp_row, conc_row, fp_col, conc_col), dim=1)
        drug_y = self.data.ddi_edge_attr[idx, 2]

        return drug_x, drug_y

    def random_split(self, test_prob, valid_prob):
        num_examples = self.data.ddi_edge_idx.shape[1]

        nvalid = int(num_examples * valid_prob)
        ntest = int(num_examples * test_prob)
        idx = torch.randperm(num_examples)

        train_idx = idx[ntest + nvalid:]
        val_idx = idx[:nvalid]
        test_idx = idx[:ntest]

        return train_idx, val_idx, test_idx

if __name__ == '__main__':

    dataset = NewDrugComb()

