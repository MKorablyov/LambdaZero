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
from scipy.linalg import eigvals


class DrugCombDb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4,
                 scores=('ZIP', 'Bliss', 'Loewe', 'HSA')):

        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        self.scores = scores

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + '/DrugCombDb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

        assert set(scores).issubset(('ZIP', 'Bliss', 'Loewe', 'HSA'))
        scores_col_idx = [['ZIP', 'Bliss', 'Loewe', 'HSA'].index(i) for i in scores]
        self.data.ddi_edge_attr = self.data.ddi_edge_attr[scores_col_idx]

        self._get_protein_laplacian_features(self.data.ppi_edge_idx)

    @property
    def raw_file_names(self):
        return [
            'drug_chemical_info.csv',
            'drugcombs_scored.csv',
            'drug_protein_links.tsv',
            'protein_protein_links.txt'
        ]

    @property
    def processed_file_names(self):
        return ['drug_combdb_data.pt']

    def download(self):
        urls = [
            'http://drugcombdb.denglab.org/download/drug_chemical_info.csv',
            'http://drugcombdb.denglab.org/download/drugcombs_scored.csv',
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
        self._drugcomb_scored = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names[1]))
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

        x = self.nodes.drop(['cIds', 'drugNameOfficial', 'molecularWeight', 'smilesString', 'name', 'is_drug'], axis=1)
        x = x.to_numpy().astype(np.int)

        data = Data(x=torch.tensor(x, dtype=torch.float))

        # Add ppi attributes to data
        data.ppi_edge_idx = torch.tensor(ppi_edge_idx, dtype=torch.long)
        data.ppi_edge_attr = torch.tensor(ppi_edge_attr, dtype=torch.long)
        # Add dpi attributes to data
        data.dpi_edge_idx = torch.tensor(dpi_edge_idx, dtype=torch.long)
        data.dpi_edge_attr = torch.tensor(dpi_edge_attr, dtype=torch.long)
        # Add ddi attributes to data
        data.ddi_edge_idx = torch.tensor(ddi_edge_idx, dtype=torch.long)
        data.ddi_edge_attr = torch.tensor(ddi_edge_attr, dtype=torch.long)
        data.ddi_edge_classes = torch.tensor(ddi_edge_classes, dtype=torch.long)

        data.is_drug = torch.tensor(self.is_drug, dtype=torch.long)
        data.number_of_drugs = self.is_drug.sum()[0]
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
        nodes = nodes.fillna(-1)

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

        protein_laplacian_features = self._get_protein_laplacian_features(ppi_edge_idx)

        return ppi_edge_idx, ppi_edge_attr

    def _get_protein_laplacian_features(self, ppi_edge_idx):
        print('Computing laplacian features of the proteins')
        # Build Laplacian matrix
        n_edges = len(ppi_edge_idx.numpy()[0]) // 2
        # Adjacency matrix
        A = coo_matrix(
            (np.ones(n_edges), (ppi_edge_idx.numpy()[0, :n_edges], ppi_edge_idx.numpy()[1, :n_edges])),
            dtype=np.int8).toarray()

        # Node degree matrix
        D = np.diag(A.sum(axis=0))

        L = D - A

        return 0

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

        self._drugcomb_scored ['idx_Drug1'] = self._drugcomb_scored['Drug1'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)
        self._drugcomb_scored['idx_Drug2'] = self._drugcomb_scored['Drug2'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        self._drugcomb_scored = self._drugcomb_scored[self._drugcomb_scored['idx_Drug1'] != -1]
        self._drugcomb_scored = self._drugcomb_scored[self._drugcomb_scored['idx_Drug2'] != -1]

        # Keep the first score for each (drug, drug, cell line) tuple that exists
        self._drugcomb_scored = self._drugcomb_scored.drop_duplicates(['idx_Drug1', 'idx_Drug2', 'Cell line'])

        # Remove edges with invalid scores
        scores = self._drugcomb_scored[self.scores]
        good_scores_arr = ((scores.notna()) & (-100 <= scores) & (scores <= 100)).all(axis=1)
        self._drugcomb_scored = self._drugcomb_scored[good_scores_arr]

        cell_lines = self._drugcomb_scored['Cell line'].unique()
        mapping = {cell_line: i for i, cell_line in enumerate(cell_lines)}
        ddi_edge_classes = self._drugcomb_scored.replace({'Cell line': mapping})['Cell line']
        ddi_edge_classes = ddi_edge_classes.to_numpy()

        ddi_edge_idx = self._drugcomb_scored[['idx_Drug1', 'idx_Drug2']].to_numpy().T
        ddi_edge_attr = self._drugcomb_scored[self.scores].to_numpy()

        return ddi_edge_idx, ddi_edge_attr, ddi_edge_classes


if __name__ == '__main__':

    dataset = DrugCombDb()

    dataset.get_edge_type('ddi')

