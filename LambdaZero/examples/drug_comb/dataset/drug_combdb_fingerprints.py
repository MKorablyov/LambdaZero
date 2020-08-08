from LambdaZero.utils import get_external_dirs
from torch.utils.data import Dataset
from torch_geometric.data import download_url
#from unrar import rarfile
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def files_exist(paths):
    return len(paths) != 0 and all([osp.exists(f) for f in paths])

class DrugCombDbFingerprints(Dataset):
    def __init__(self, transform=None, fp_bits=1024, fp_radius=4):
        super().__init__()

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.transform = transform

        datasets_dir, _, _ = get_external_dirs()
        self.root = osp.expanduser(osp.normpath(datasets_dir)) + "/%s" % type(self).__name__
        if not osp.exists(self.root):
            makedirs(self.root)

        if not osp.exists(self.processed_path):
            self._process()

        self.ddi_edge_idx, self.ddi_edge_attr, self.ddi_edge_trgt, self.fingerprints = \
            torch.load(self.processed_path)

    def to(self, device):
        self.ddi_edge_idx = self.ddi_edge_idx.to(device)
        self.ddi_edge_attr = self.ddi_edge_attr.to(device)
        self.ddi_edge_trgt = self.ddi_edge_trgt.to(device)
        self.fingerprints = self.fingerprints.to(device)

        return self

    def __len__(self):
        return self.ddi_edge_idx.shape[1]

    def __getitem__(self, idx):
        """
        Returns
        -------
        Tuple of different values.
        drug_row : torch.LongTensor
            Tensor of shape (idx.shape[0],) where each item is the row index
            of the drug pair.
        drug_col : torch.LongTensor
            Tensor of shape (idx.shape[0],) where each item is the column index
            of the drug pair.
        conc_row : torch.FloatTensor
            Tensor of shape (idx.shape[0],) where each item is the concentration
            of the row drug for the drug pair.
        conc_col : torch.FloatTensor
            Tensor of shape (idx.shape[0],) where each item is the concentration
            of the column drug for the drug pair.
        y : torch.FloatTensor
            Tensor of shape (idx.shape[0],) where each item is the response for
            the drug pair.
        """
        drug_row = self.ddi_edge_idx[0, idx]
        drug_col = self.ddi_edge_idx[1, idx]
        conc_row = self.ddi_edge_attr[0, idx]
        conc_col = self.ddi_edge_attr[1, idx]
        y = self.ddi_edge_trgt[idx]

        if self.transform is not None:
            drug_row, drug_col, conc_row, conc_col, y = \
                self.transform(drug_row, drug_col, conc_row, conc_col, y)

        return drug_row, drug_col, conc_row, conc_col, y

    @property
    def processed_path(self):
        return osp.join(self.root, 'fp_ddi_data.pt')

    @property
    def raw_paths(self):
        return [
            osp.join(self.root, 'drug_chemical_info.csv'),
            osp.join(self.root, 'drugcombs_response.csv'),
            osp.join(self.root, 'drug_protein_links.tsv'),
        ]

    def _download(self):
        #urls = [
        #    'http://drugcombdb.denglab.org/download/drug_chemical_info.csv',
        #    'http://drugcombdb.denglab.org/download/drugcombs_response.csv',
        #    'http://drugcombdb.denglab.org/download/drug_protein_links.rar',
        #]

        #for url in urls:
        #    download_url(url, self.root)

        rar = rarfile.RarFile(self.raw_paths[2])
        rar.extractall(path=self.root)

    def _process(self):
        if not files_exist(self.raw_paths):
            self._download()

        # Load dataframes
        self._drug_chemical_info = pd.read_csv(self.raw_paths[0], encoding='ISO-8859-1')
        self._drugcomb_response = pd.read_csv(self.raw_paths[1])
        self._drug_protein_link = pd.read_csv(self.raw_paths[2], sep='\t')

        drug_nodes = self._get_drug_nodes()

        # Build dictionaries necessary to build edges
        name_to_idx_dict = {drug_nodes.at[i, 'name']: i for i in range(len(drug_nodes))}

        # Build edges
        ddi_edge_idx, ddi_edge_attr, ddi_edge_trgt = self._get_ddi_edges(name_to_idx_dict)  # One way only

        # Compute drug and protein feature arrays
        x_drugs = drug_nodes.drop(['cIds', 'drugNameOfficial', 'molecularWeight',
                                   'smilesString', 'name', 'is_drug'], axis=1)

        ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.float)
        ddi_edge_trgt = torch.tensor(ddi_edge_trgt, dtype=torch.float)
        fingerprints = torch.tensor(x_drugs.to_numpy(), dtype=torch.float)

        torch.save((ddi_edge_idx, ddi_edge_attr, ddi_edge_trgt, fingerprints), self.processed_path)

    def _get_fingerprint(self, smile, radius, n_bits):
        if smile == 'none':
            return np.array([-1] * n_bits)
        try:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
        except Exception as ex:
            return np.array([-1] * n_bits)

    def _get_drug_nodes(self):
        print('Processing drug nodes..')

        # Computing fingerprints
        all_fp = self._drug_chemical_info['smilesString'].apply(
            lambda s: self._get_fingerprint(s, radius=self.fp_radius, n_bits=self.fp_bits)
        )

        # Convert fingerprints to dataframe
        all_fp = list(all_fp)
        all_fp = [list(fp) for fp in all_fp]
        all_fp = pd.DataFrame(all_fp, columns=["fp" + str(i) for i in range(self.fp_bits)])

        # Add to drug info
        drug_nodes = pd.concat((self._drug_chemical_info, all_fp), axis=1)

        # drop drugs without fingerprint
        drug_nodes = drug_nodes[drug_nodes['fp0'].apply(lambda fp: fp != -1)]

        # drop drugs without protein targets
        drugs_with_targets = set(self._drug_protein_link['chemical'])
        drug_nodes = drug_nodes[drug_nodes['cIds'].apply(lambda s: s in drugs_with_targets)]

        # Modifications for compatibility with protein nodes
        drug_nodes = drug_nodes.rename(columns={'drugName': 'name'})
        drug_nodes['is_drug'] = 1

        return drug_nodes.reset_index(drop=True)

    def _get_ddi_edges(self, name_to_idx_dict):
        print('Processing drug drug interaction edges..')

        # Remove rows with nan values
        self._drugcomb_response = self._drugcomb_response.dropna(axis=0)

        self._drugcomb_response['idx_DrugRow'] = self._drugcomb_response['DrugRow'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        self._drugcomb_response['idx_DrugCol'] = self._drugcomb_response['DrugCol'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        self._drugcomb_response = self._drugcomb_response[self._drugcomb_response['idx_DrugRow'] != -1]
        self._drugcomb_response = self._drugcomb_response[self._drugcomb_response['idx_DrugCol'] != -1]

        # Remove combinations of a drug with itself
        is_pairs_of_diff_drugs = self._drugcomb_response[['idx_DrugRow', 'idx_DrugCol']].apply(
            lambda s: s['idx_DrugRow'] != s['idx_DrugCol'], axis=1)
        self._drugcomb_response = self._drugcomb_response[is_pairs_of_diff_drugs]

        ddi_edge_idx = self._drugcomb_response[['idx_DrugRow', 'idx_DrugCol']].to_numpy().T
        ddi_edge_attr = self._drugcomb_response[['ConcRow', 'ConcCol']].to_numpy().T
        ddi_edge_trgt = self._drugcomb_response['Response'].to_numpy()

        ddi_edge_attr[0, :] = np.log(ddi_edge_attr[0, :] + 1e-6)
        ddi_edge_attr[1, :] = np.log(ddi_edge_attr[1, :] + 1e-6)

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_trgt

    def random_split(self, test_prob, valid_prob):
        num_examples = len(self)

        nvalid = int(num_examples * valid_prob)
        ntest = int(num_examples * test_prob)
        idx = torch.randperm(num_examples)

        train_idx = idx[ntest + nvalid:]
        val_idx = idx[:nvalid]
        test_idx = idx[:ntest]

        return train_idx, val_idx, test_idx


if __name__ == '__main__':
    dataset = DrugCombDbFingerprints()

