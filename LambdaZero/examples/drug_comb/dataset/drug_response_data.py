from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Data, InMemoryDataset, download_url
from unrar import rarfile
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import torch
import os
from scipy.sparse import coo_matrix


class DrugResponse(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4, n_laplace_feat=256, **kwargs):

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.n_laplace_feat = n_laplace_feat

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + '/DrugCombDb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        # Log transform concentrations
        self.data.ddi_edge_attr[:, 1] = torch.log(self.data.ddi_edge_attr[:, 1] + 1e-6)
        self.data.ddi_edge_attr[:, 2] = torch.log(self.data.ddi_edge_attr[:, 2] + 1e-6)

    @property
    def raw_file_names(self):
        return [
            'drug_chemical_info.csv',
            'drugcombs_response.csv',
            'drug_protein_links.tsv',
            'protein_protein_links.txt'
        ]

    @property
    def processed_file_names(self):
        return ['drug_response_data.pt']

    def download(self):
        urls = [
            'http://drugcombdb.denglab.org/download/drug_chemical_info.csv',
            'http://drugcombdb.denglab.org/download/drugcombs_response.csv',
            'http://drugcombdb.denglab.org/download/drug_protein_links.rar',
            'http://drugcombdb.denglab.org/download/protein_protein_links.rar'
        ]

        for url in urls:
            download_url(url, self.raw_dir)

        rar_filenames = ['/drug_protein_links.rar', '/protein_protein_links.rar']
        for rar_filename in rar_filenames:
            rar = rarfile.RarFile(self.raw_dir + rar_filename)
            rar.extractall(path=self.raw_dir)

    def process(self):

        # Load dataframes
        self._drug_chemical_info = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[0]),
                                               encoding='ISO-8859-1')
        self._drugcomb_response = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]))
        self._drug_protein_link = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[2]), sep='\t')
        self._protein_protein_interactions = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[3]), sep=' ')

        self.nodes, self.is_drug = self._get_nodes()

        # Build dictionaries necessary to build edges
        cid_to_idx_dict = {self.nodes.at[i, 'cIds']: i for i in range(len(self.nodes))}
        name_to_idx_dict = {self.nodes.at[i, 'name']: i for i in range(len(self.nodes))}

        # Build edges
        ppi_edge_idx, ppi_edge_attr = self._get_ppi_edges(name_to_idx_dict)  # PPI edges are both ways
        dpi_edge_idx, dpi_edge_attr = self._get_dpi_edges(cid_to_idx_dict, name_to_idx_dict)  # One way only
        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes = self._get_ddi_edges(name_to_idx_dict)  # One way only

        # Compute protein laplacian features
        protein_laplacian_features = self._get_protein_laplacian_features(ppi_edge_idx, self.n_laplace_feat)

        # Compute drug and protein feature arrays
        x_drugs = self.nodes.drop(['cIds', 'drugNameOfficial', 'molecularWeight', 'smilesString', 'name', 'is_drug'],
                                  axis=1)
        x_drugs = x_drugs.to_numpy().astype(np.int)
        x_prots = protein_laplacian_features.to_numpy()

        data = Data(x_drugs=torch.tensor(x_drugs, dtype=torch.float), x_prots=torch.tensor(x_prots, dtype=torch.float))

        # Add ppi attributes to data
        data.ppi_edge_idx = torch.tensor(ppi_edge_idx, dtype=torch.long)
        data.ppi_edge_attr = torch.tensor(ppi_edge_attr, dtype=torch.float)
        # Add dpi attributes to data
        data.dpi_edge_idx = torch.tensor(dpi_edge_idx, dtype=torch.long)
        data.dpi_edge_attr = torch.tensor(dpi_edge_attr, dtype=torch.float)
        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.float)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.float)

        data.is_drug = torch.tensor(self.is_drug, dtype=torch.long)
        data.number_of_drugs = self.is_drug.sum()
        data.number_of_proteins = len(self.is_drug) - data.number_of_drugs

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_fingerprint(self, smile, radius, n_bits):
        if smile == 'none':
            return np.array([-1] * n_bits)
        try:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
        except Exception as ex:
            return np.array([-1] * n_bits)

    def _get_nodes(self):
        drug_nodes = self._get_drug_nodes()
        protein_nodes = self._get_protein_nodes()

        nodes = pd.concat((drug_nodes, protein_nodes), ignore_index=True, sort=False)
        nodes = nodes.fillna(0)

        is_drug = nodes['is_drug'].to_numpy()

        return nodes, is_drug

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

    def _get_protein_nodes(self):
        print('Processing protein nodes..')

        all_proteins = set(self._drug_protein_link['protein']).union(
            set(self._protein_protein_interactions['protein1'])
        ).union(set(self._protein_protein_interactions['protein2']))

        protein_nodes = pd.DataFrame(all_proteins, columns=['name'])
        protein_nodes['is_drug'] = 0

        return protein_nodes

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

    def _get_protein_laplacian_features(self, ppi_edge_idx, n_features=256):

        if not os.path.isfile(os.path.join(self.raw_dir, "laplacian_eigvects.npz")):
            print('Computing laplacian features of the proteins, only happens the first time')
            # Build Laplacian matrix
            n_edges = len(ppi_edge_idx[0]) // 2
            # Adjacency matrix
            A = coo_matrix(
                (np.ones(n_edges), (ppi_edge_idx[0, :n_edges],
                                    ppi_edge_idx[1, :n_edges])),
                dtype=np.int8).toarray()

            # Node degree matrix
            D = np.diag(A.sum(axis=0))
            L = D - A

            # Restrict to protein nodes
            L = L[self.is_drug.sum():, self.is_drug.sum():]

            eigvals, eigvects = np.linalg.eigh(L)  # Diagonalize the matrix
            # Save results
            np.savez(os.path.join(self.raw_dir, "laplacian_eigvects.npz"), eigvals=eigvals, eigvects=eigvects)

        else:
            npzfile = np.load(os.path.join(self.raw_dir, "laplacian_eigvects.npz"))
            eigvals, eigvects = npzfile['eigvals'], npzfile['eigvects']

        # Remove zero eigvalues and corresponding eigvectors
        nonzero_vals = eigvals > 1e-6
        eigvects = eigvects[:, nonzero_vals]

        # Transform to dataframe
        n_drugs = self.is_drug.sum()
        col_names = ["lap_feat_" + str(i) for i in range(256)]
        zeros_for_drugs = pd.DataFrame(0, index=np.arange(n_drugs), columns=col_names)
        laplacian_feats = pd.DataFrame(eigvects[:, :n_features], columns=col_names)

        return pd.concat((zeros_for_drugs, laplacian_feats), ignore_index=True)

    def _get_dpi_edges(self, cid_to_idx_dict, name_to_idx_dict):
        print('Processing drug protein edges..')

        self._drug_protein_link['idx_chemical'] = self._drug_protein_link['chemical'].apply(
            lambda s: cid_to_idx_dict[s] if s in cid_to_idx_dict.keys() else -1)

        self._drug_protein_link['idx_prot'] = \
            self._drug_protein_link['protein'].apply(lambda s: name_to_idx_dict[s])

        # Remove items without chem info
        dp_links_with_info = self._drug_protein_link[self._drug_protein_link['idx_chemical'] != -1]

        dpi_edge_idx = dp_links_with_info[['idx_chemical', 'idx_prot']].to_numpy().T
        dpi_edge_attr = dp_links_with_info['combined_score'].to_numpy()

        return dpi_edge_idx, dpi_edge_attr

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
        ddi_edge_classes = np.zeros((len(self._drugcomb_response)))  # For compatibility between datasets
        ddi_edge_attr = self._drugcomb_response[['Response', 'ConcRow', 'ConcCol']].to_numpy()

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes

    def random_split(self, test_prob, valid_prob):
        """
        We split at the edge index level. the data.ddi_edge_idx attribute contains the "same" edge several times, each
        time corresponding to a specific pair of concentrations for each drug.

        We split so that all edges that correspond to the same drug pair end up in the same split
        """

        def get_set_from_idx(idx, i):
            """
            return 0 if unique edge i is in train, 1 if valid, 2 if test
            """
            if idx[i] >= ntest + nvalid:
                return 0
            elif idx[i] < nvalid:
                return 1
            else:
                return 2

        # Get unique edges (one for each drug pair, regardless of concentrations)
        unique_ddi_edge_idx = self.data.ddi_edge_idx.unique(dim=1)
        num_unique_examples = unique_ddi_edge_idx.shape[1]

        # train test valid split of unique edges
        nvalid = int(num_unique_examples * valid_prob)
        ntest = int(num_unique_examples * test_prob)
        unique_idx = torch.randperm(num_unique_examples)

        # Dictionary that associate each unique edge with a split (train valid or test)
        edge_to_split_dict = {tuple(unique_ddi_edge_idx.T[i].tolist()): get_set_from_idx(unique_idx, i)
                              for i in range(num_unique_examples)}

        # Split for all edges (non unique)
        all_edges_split = np.array([edge_to_split_dict[tuple(edge.tolist())] for edge in self.data.ddi_edge_idx.T])

        train_idx = np.where(all_edges_split == 0)[0]
        val_idx = np.where(all_edges_split == 1)[0]
        test_idx = np.where(all_edges_split == 2)[0]

        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        return torch.tensor(train_idx), torch.tensor(val_idx), torch.tensor(test_idx)


if __name__ == '__main__':

    dataset = DrugResponse()

