from torch_geometric.data import InMemoryDataset, download_url
from LambdaZero.examples.drug_comb.utils import get_fingerprint
from LambdaZero.examples.drug_comb.utils import get_external_dirs
from pubchempy import Compound
import urllib.request
from unrar import rarfile
import ssl
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import random


########################################################################################################################
# Abstract DrugComb dataset
########################################################################################################################


class AbstractDrugComb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, ppi_confidence_thres=0):
        """
        Dataset object for the drugcomb dataset. The drug responses are retrieved from https://drugcomb.fimm.fi
        PPI and DPI graphs are retrieved from http://drugcombdb.denglab.org
        All data will be downloaded and processed on the first instantiation.
        :param transform:
        :param pre_transform:
        :param fp_bits: number of bits for fingerprints
        :param fp_radius: radius for fingerprints
        :param ppi_confidence_thres: remove ppi edges whose confidence is lower that this threshold
        """

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(os.path.join(datasets_dir, 'DrugComb/'), transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        # If the fingerprint parameters do not correspond, reprocess dataset
        if self.data.fp_bits != self.fp_bits or self.data.fp_radius != self.fp_radius:
            self.data, self.slices = 0, 0
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])

        # Filter PPI edges that are below the confidence threshold
        if self.data.ppi_edge_idx.shape[1] > 0:
            self.data.ppi_edge_idx = self.data.ppi_edge_idx[:, self.data.ppi_edge_attr > ppi_confidence_thres]
            self.data.ppi_edge_attr = self.data.ppi_edge_attr[self.data.ppi_edge_attr > ppi_confidence_thres]

        print("Dataset loaded.")
        print("\t", self.data.ddi_edge_idx.shape[1], "drug comb experiments among", self.data.x_drugs.shape[0], "drugs")
        print("\t fingeprints with radius", self.fp_radius, "and nbits", self.fp_bits)
        print("\t", self.data.dpi_edge_idx.shape[1], "drug target interactions")
        print("\t", int(self.data.ppi_edge_idx.shape[1]/2), "prot-prot interactions")

    def process(self):
        raise NotImplementedError

    @property
    def raw_file_names(self):
        return [
            'drugcomb_data_v1.4.csv',
            'summary_table_v1.4.csv',
            'drug_protein_links.tsv',
            'protein_protein_links.txt'
        ]

    def download(self):
        urls = [
            'https://drugcomb.fimm.fi/download/drugcomb_data_v1.4.csv',
            'https://drugcomb.fimm.fi/download/summary_table_v1.4.csv',
            'http://drugcombdb.denglab.org/download/drug_protein_links.rar',
            'http://drugcombdb.denglab.org/download/protein_protein_links.rar']

        for url in urls[2:]:
            download_url(url, self.raw_dir)

        rar_filenames = ['/drug_protein_links.rar', '/protein_protein_links.rar']
        for rar_filename in rar_filenames:
            rar = rarfile.RarFile(self.raw_dir + rar_filename)
            rar.extractall(path=self.raw_dir)

        chunk_size = 16 * 1024
        for url in urls[:2]:
            file_name = url.split('/')[-1]
            if os.path.isfile(os.path.join(self.raw_dir, file_name)):
                print('Using exist file', file_name)
            else:
                print('Downloading', url)
                gcontext = ssl.SSLContext()
                with urllib.request.urlopen(url, context=gcontext) as f:
                    with open(os.path.join(self.raw_dir, file_name), "w") as file:
                        while True:
                            file_chunk = f.read(chunk_size).decode('utf-8')
                            if not file_chunk:
                                break
                            file.write(file_chunk)

    def random_split(self, test_prob, valid_prob):
        """
        We split at the edge index level. the data.ddi_edge_idx attribute contains the "same" edge several times, each
        time corresponding to a specific pair of concentrations for each drug.

        We split so that all edges that correspond to the same drug pair end up in the same split
        """

        # Get unique edges (one for each drug pair, regardless of cell line)
        unique_ddi_edge_idx = self.data.ddi_edge_idx.unique(dim=1)
        num_unique_examples = unique_ddi_edge_idx.shape[1]

        # train test valid split of unique edges
        nvalid = int(num_unique_examples * valid_prob)
        ntest = int(num_unique_examples * test_prob)

        # Shuffle with seed 0
        shuflled_idx = np.arange(num_unique_examples)
        random.Random(0).shuffle(shuflled_idx)

        unique_test_idx = shuflled_idx[:ntest]  # Fixed test set
        unique_train_valid_idx = shuflled_idx[ntest:]  # Train and valid sets
        # Shuffle train and valid with current seed
        unique_train_valid_idx = unique_train_valid_idx[torch.randperm(len(unique_train_valid_idx))]
        # Get train and valid
        unique_valid_idx = unique_train_valid_idx[:nvalid]
        unique_train_idx = unique_train_valid_idx[nvalid:]

        # Dictionary that associate each unique edge with a split (train: 0, valid: 1, test: 2)
        edge_to_split_dict = {**{tuple(unique_ddi_edge_idx.T[i].tolist()): 0
                              for i in unique_train_idx},
                              **{tuple(unique_ddi_edge_idx.T[i].tolist()): 1
                              for i in unique_valid_idx},
                              **{tuple(unique_ddi_edge_idx.T[i].tolist()): 2
                              for i in unique_test_idx}}

        # Split for all edges (non unique)
        all_edges_split = np.array([edge_to_split_dict[tuple(edge.tolist())] for edge in self.data.ddi_edge_idx.T])

        train_idx = np.where(all_edges_split == 0)[0]
        val_idx = np.where(all_edges_split == 1)[0]
        test_idx = np.where(all_edges_split == 2)[0]

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        return torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)

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

    def _get_protein_nodes(self):
        print('Processing protein nodes..')

        all_proteins = set(self._drug_protein_link['protein']).union(
            set(self._protein_protein_interactions['protein1'])
        ).union(set(self._protein_protein_interactions['protein2']))

        protein_nodes = pd.DataFrame(all_proteins, columns=['name'])

        return protein_nodes

    def _get_drug_nodes(self):
        print('Processing drug nodes..')

        self._drugcomb_data['drug_row_smiles'] = self._drugcomb_data['drug_row_cid'].apply(
            lambda cid: self.cid_to_smiles_dict[cid] if cid in self.cid_to_smiles_dict.keys() else -1)

        self._drugcomb_data['drug_col_smiles'] = self._drugcomb_data['drug_col_cid'].apply(
            lambda cid: self.cid_to_smiles_dict[cid] if cid in self.cid_to_smiles_dict.keys() else -1)

        # Computing fingerprints
        cid_to_fp_dict = {cid: get_fingerprint(self.cid_to_smiles_dict[cid], self.fp_radius, self.fp_bits)
                          for cid in self.cid_to_smiles_dict.keys()}

        drug_nodes = pd.DataFrame.from_dict(cid_to_fp_dict, orient='index')

        # drop drugs without protein targets
        drugs_with_targets = set(self._drug_protein_link['chemical'].apply(lambda s: int(s[4:])))
        drug_nodes = drug_nodes[pd.Series(drug_nodes.index, index=drug_nodes.index).
            apply(lambda s: s in drugs_with_targets)]

        return drug_nodes

    def _get_nodes(self):
        drug_nodes = self._get_drug_nodes()
        protein_nodes = self._get_protein_nodes()

        cid_to_idx_dict = {drug_nodes.index[i]: i for i in range(len(drug_nodes))}
        drug_nodes.reset_index(drop=True, inplace=True)

        protein_nodes.index += len(drug_nodes)
        name_to_idx_dict = {protein_nodes.at[i, 'name']: i for i in protein_nodes.index}

        protein_nodes['index'] = protein_nodes.index
        protein_nodes.drop('name', axis=1, inplace=True)

        is_drug = np.array([1] * drug_nodes.shape[0] + [0] * protein_nodes.shape[0])[:, None]

        return drug_nodes, protein_nodes, cid_to_idx_dict, name_to_idx_dict, is_drug

    def _get_dpi_edges(self, cid_to_idx_dict, name_to_idx_dict):
        print('Processing drug protein edges..')

        self._drug_protein_link['idx_chemical'] = self._drug_protein_link['chemical'].apply(
            lambda s: cid_to_idx_dict[int(s[4:])] if int(s[4:]) in cid_to_idx_dict.keys() else -1)

        self._drug_protein_link['idx_prot'] = \
            self._drug_protein_link['protein'].apply(lambda s: name_to_idx_dict[s])

        # Remove items without chem info
        dp_links_with_info = self._drug_protein_link[self._drug_protein_link['idx_chemical'] != -1]

        dpi_edge_idx = dp_links_with_info[['idx_chemical', 'idx_prot']].to_numpy().T
        dpi_edge_attr = dp_links_with_info['combined_score'].to_numpy()

        return dpi_edge_idx, dpi_edge_attr

    def _get_ppi_edges(self, name_to_idx_dict):
        print('Processing protein protein edges..')

        self._protein_protein_interactions['idx_prot1'] = \
            self._protein_protein_interactions['protein1'].apply(lambda s: name_to_idx_dict[s])

        self._protein_protein_interactions['idx_prot2'] = \
            self._protein_protein_interactions['protein2'].apply(lambda s: name_to_idx_dict[s])

        ppi_edge_idx = self._protein_protein_interactions[['idx_prot1', 'idx_prot2']].to_numpy().T
        ppi_edge_attr = self._protein_protein_interactions['combined_score'].to_numpy()

        # Edges are directed, we need to feed them both ways
        ppi_edge_idx = np.concatenate((ppi_edge_idx, ppi_edge_idx[::-1, :]), axis=1)
        ppi_edge_attr = np.concatenate((ppi_edge_attr, ppi_edge_attr), axis=0)

        return ppi_edge_idx, ppi_edge_attr
