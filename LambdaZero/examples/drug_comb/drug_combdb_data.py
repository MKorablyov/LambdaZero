from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Data, InMemoryDataset, download_url
from unrar import rarfile
import numpy as np
import pandas as pd
import torch

def get_fingerprint(smile, radius, n_bits):
    if smile == 'none':
        return np.array([-1]*n_bits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
    except:
        return np.array([-1]*n_bits)

def transform_to_bipartite_drug_protein_graph(data_list):
    new_data_list = []
    for data in data_list:
        dpi_edge_idx_range = data.graph_type_idx_ranges['dpi']

        new_edge_idx_first = data.edge_index[:, dpi_edge_idx_range[0]:dpi_edge_idx_range[1]]
        new_edge_attr_first = data.edge_attr[dpi_edge_idx_range[0]:dpi_edge_idx_range[1]]

        new_edge_idx_scnd = data.edge_index[:, dpi_edge_idx_range[0] * 2:dpi_edge_idx_range[1] * 2]
        new_edge_attr_scnd = data.edge_attr[dpi_edge_idx_range[0] * 2:dpi_edge_idx_range[1] * 2]

        new_edge_idx = np.concatenate((new_edge_idx_first, new_edge_idx_scnd), axis=1)
        new_edge_attr = np.concatenate((new_edge_attr_first, new_edge_attr_scnd), axis=0)

        new_data_list.append(Data(
            x=data.x,
            edge_index=torch.tensor(new_edge_idx, dtype=torch.long),
            edge_attr=new_edge_attr
        ))

    return new_data_list

class DrugCombDb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        self._drug_protein_link_holder = None
        self._protein_protein_interactions_holder = None

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + '/DrugCombDb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def _drug_protein_link(self):
        if self._drug_protein_link_holder is None:
            dpi_filename = '%s/%s' % (self.raw_dir, self.raw_file_names[2])
            self._drug_protein_link_holder = pd.read_csv(dpi_filename, sep='\t', encoding='ISO-8859-1')

        return self._drug_protein_link_holder

    @property
    def _protein_protein_interactions(self):
        if self._protein_protein_interactions_holder is None:
            ppi_filename = '%s/%s' % (self.raw_dir, self.raw_file_names[3])
            self._protein_protein_interactions_holder = pd.read_csv(ppi_filename, sep=' ', encoding='ISO-8859-1')

        return self._protein_protein_interactions_holder

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
            'http://drugcombdb.denglab.org/download/drugcombs_scored.csv',
            'http://drugcombdb.denglab.org/download/drug_protein_links.rar',
            'http://drugcombdb.denglab.org/download/protein_protein_links.rar',
            'http://drugcombdb.denglab.org/download/drug_chemical_info.csv'
        ]

        for url in urls:
            download_url(url, self.raw_dir)

        rar_filenames = ['/drug_protein_links.rar', '/protein_protein_links.rar']
        for rar_filename in rar_filenames:
            rar = rarfile.RarFile(self.raw_dir + rar_filename)
            rar.extractall(path=self.raw_dir)

    def process(self):
        nodes = self._get_nodes()
        edge_idxs, edge_attr, edge_classes, graph_type_idx_ranges = self._get_edges_information(nodes)

        data = Data(
            x=torch.tensor(self._get_node_ftrs(nodes), dtype=torch.float),
            edge_index=torch.tensor(edge_idxs, dtype=torch.long),
            edge_attr=edge_attr,
        )

        data.edge_classes = edge_classes
        data.graph_type_idx_ranges = graph_type_idx_ranges

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_node_ftrs(self, nodes):
        node_ftrs = nodes.fillna(-1)
        node_ftrs = node_ftrs.drop(
            ['cIds', 'drugNameOfficial', 'molecularWeight', 'smilesString', 'name'], axis=1)

        return node_ftrs.to_numpy().astype(np.int)

    def _get_nodes(self):
        drug_nodes = self._get_drug_nodes()
        protein_nodes = self._get_protein_nodes()

        nodes = pd.concat((drug_nodes, protein_nodes), ignore_index=True, sort=False)

        return nodes

    def _get_drug_nodes(self):
        print('Processing drug nodes..')

        drug_info_filename = '%s/%s' % (self.raw_dir, self.raw_file_names[0])
        drug_chem_info_no_fp = pd.read_csv(drug_info_filename, encoding='ISO-8859-1')
        drug_chem_info_with_fp = self._augment_drug_info_with_fp(drug_chem_info_no_fp)

        drug_chem_info_with_fp['has_fp'] = drug_chem_info_with_fp['fp0'].apply(lambda fp: fp != -1)
        drug_chem_info_with_fp['is_drug'] = 1
        drug_nodes = drug_chem_info_with_fp.rename(columns={'drugName': 'name'})

        return drug_nodes

    def _augment_drug_info_with_fp(self, drug_chem_info_no_fp):
        n_bits = 1024
        radius = 4

        all_fp = drug_chem_info_no_fp['smilesString'].apply(
            lambda s: get_fingerprint(s, radius=radius, n_bits=n_bits)
        )

        # Convert to dataframe
        all_fp = list(all_fp)
        all_fp = [list(fp) for fp in all_fp]
        all_fp = pd.DataFrame(all_fp, columns=["fp" + str(i) for i in range(n_bits)])

        return pd.concat((drug_chem_info_no_fp, all_fp), axis=1)

    def _get_protein_nodes(self):
        print('Processing protein nodes..')

        all_proteins = set(self._drug_protein_link['protein']).union(
            set(self._protein_protein_interactions['protein1'])
        ).union(set(self._protein_protein_interactions['protein2']))

        protein_nodes = pd.DataFrame(all_proteins, columns=['name'])
        protein_nodes['is_drug'] = 0
        protein_nodes['has_fp'] = False

        return protein_nodes

    def _get_edges_information(self, nodes):
        print('Processing edges..')

        cid_to_idx_dict = {nodes.at[i, 'cIds']: i for i in range(len(nodes))}
        name_to_idx_dict = {nodes.at[i, 'name']: i for i in range(len(nodes))}

        ppi_edge_idx, ppi_edge_attr, ppi_edge_classes = self._get_ppi_edges(name_to_idx_dict)
        dpi_edge_idx, dpi_edge_attr, dpi_edge_classes = self._get_dpi_edges(cid_to_idx_dict, name_to_idx_dict)
        ddi_edge_idx, ddi_edge_attr, ddi_edge_classes = self._get_ddi_edges(name_to_idx_dict)

        all_edge_idxs = np.concatenate((ppi_edge_idx, dpi_edge_idx, ddi_edge_idx), axis=1)
        all_edge_attr = np.concatenate((ppi_edge_attr, dpi_edge_attr, ddi_edge_attr), axis=0)
        all_edge_classes = np.concatenate((ppi_edge_classes, dpi_edge_classes, ddi_edge_classes))

        # Edges are directed, we need to feed them both ways
        all_edge_idxs = np.concatenate((all_edge_idxs, all_edge_idxs[::-1, :]), axis=1)
        all_edge_attr = np.concatenate((all_edge_attr, all_edge_attr), axis=0)
        all_edge_classes = np.concatenate((all_edge_classes, all_edge_classes))

        graph_type_idx_ranges = self._get_graph_type_idx_ranges(ppi_edge_idx, dpi_edge_idx, ddi_edge_idx)

        return all_edge_idxs, all_edge_attr, all_edge_classes, graph_type_idx_ranges

    def _get_ddi_edges(self, name_to_idx_dict):
        print('Processing drug drug interaction edges..')

        drug_scores_filename = '%s/%s' % (self.raw_dir, self.raw_file_names[1])
        drug_comb_scored = pd.read_csv(drug_scores_filename, encoding='ISO-8859-1')

        drug_comb_scored['idx_Drug1'] = drug_comb_scored['Drug1'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)
        drug_comb_scored['idx_Drug2'] = drug_comb_scored['Drug2'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        drug_drug_edges = drug_comb_scored[drug_comb_scored['idx_Drug1'] != -1]
        drug_drug_edges = drug_drug_edges[drug_drug_edges['idx_Drug2'] != -1]

        cell_lines = drug_drug_edges['Cell line'].unique()
        mapping = {cell_line: i for i, cell_line in enumerate(cell_lines)}
        ddi_edge_classes = drug_drug_edges.replace({'Cell line': mapping})['Cell line']
        ddi_edge_classes = ddi_edge_classes.to_numpy()

        ddi_edge_idxs = drug_drug_edges[['idx_Drug1', 'idx_Drug2']].to_numpy().T
        ddi_edge_attr = drug_drug_edges[['ZIP', 'Bliss', 'Loewe', 'HSA']].to_numpy()

        return ddi_edge_idxs, ddi_edge_attr, ddi_edge_classes

    def _get_dpi_edges(self, cid_to_idx_dict, name_to_idx_dict):
        print('Processing drug protein edges..')

        self._drug_protein_link['idx_chemical'] = self._drug_protein_link['chemical'].apply(
            lambda s: cid_to_idx_dict[s] if s in cid_to_idx_dict.keys() else -1)

        self._drug_protein_link['idx_prot'] = \
            self._drug_protein_link['protein'].apply(lambda s: name_to_idx_dict[s])

        # Remove items without chem info
        links_all_chem_info = self._drug_protein_link[self._drug_protein_link['idx_chemical'] != -1]

        dpi_edge_idxs = links_all_chem_info[['idx_chemical', 'idx_prot']].to_numpy().T
        dpi_edge_attr = np.zeros((dpi_edge_idxs.shape[1], 4))
        dpi_edge_classes = np.full((dpi_edge_idxs.shape[1]), -1)

        return dpi_edge_idxs, dpi_edge_attr, dpi_edge_classes

    def _get_ppi_edges(self, name_to_idx_dict):
        print('Processing protein protein edges..')

        self._protein_protein_interactions['idx_prot1'] = \
            self._protein_protein_interactions['protein1'].apply(lambda s: name_to_idx_dict[s])

        self._protein_protein_interactions['idx_prot2'] = \
            self._protein_protein_interactions['protein2'].apply(lambda s: name_to_idx_dict[s])

        ppi_edge_idxs = self._protein_protein_interactions[['idx_prot1', 'idx_prot2']].to_numpy().T
        ppi_edge_attr = np.zeros((ppi_edge_idxs.shape[1], 4))
        ppi_edge_classes = np.full((ppi_edge_idxs.shape[1]), -1)

        return ppi_edge_idxs, ppi_edge_attr, ppi_edge_classes

    def _get_graph_type_idx_ranges(self, ppi_edge_idx, dpi_edge_idx, ddi_edge_idx):
        graph_type_idx_ranges = {}

        graph_type_idx_ranges['ppi'] = (0, ppi_edge_idx.shape[1])
        graph_type_idx_ranges['dpi'] = (
            graph_type_idx_ranges['ppi'][1],
            graph_type_idx_ranges['ppi'][1] + dpi_edge_idx.shape[1]
        )
        graph_type_idx_ranges['ddi']  = (
            graph_type_idx_ranges['dpi'][1],
            graph_type_idx_ranges['dpi'][1] + ddi_edge_idx.shape[1]
        )

        return graph_type_idx_ranges

if __name__ == '__main__':
    dataset = DrugCombDb(pre_transform=transform_to_bipartite_drug_protein_graph)
    import pdb; pdb.set_trace()

