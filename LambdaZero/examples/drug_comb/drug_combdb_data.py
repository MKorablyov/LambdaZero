from LambdaZero.utils import get_external_dirs
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
import torch
import patool

def get_fingerprint(smile, radius, nBits):
    if smile == 'none':
        return np.array([-1]*nBits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, nBits))
    except:
        return np.array([-1]*nBits)

def transform_to_bipartite_drug_protein_graph(data):
    dpi_edge_idx_range = data.graph_type_idx_ranges['dpi']

    new_edge_idx = data.edge_index[*dpi_edge_idx_range]
    new_edge_attr = data.edge_attr[*dpi_edge_idx_range]

    return Data(x=data.x, edge_index=new_edge_idx, edge_attr=new_edge_attr)

class DrugCombDb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        self._drug_protein_link_holder = None
        self._protein_protein_interactions_holder = None

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + 'DrugCombDb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def _drug_protein_link(self):
        if self._drug_protein_link_holder is None:
            dpi_filename = self.raw_dir + self.raw_file_names[2]
            self._drug_protein_link_holder = pd.read_csv(dpi_filename, sep='\t')

        return self._drug_protein_link_holder

    @property
    def _protein_protein_interactions(self):
        if self._protein_protein_interactions_holder:
            ppi_filename = self.raw_dir + self.raw_file_names[3]
            self._protein_protein_interactions_holder = pd.read_csv(ppi_filename, sep=' ')

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

    def download(self, root_dir):
        urls = [
            'http://drugcombdb.denglab.org/download/drugcombs_scored.csv',
            'http://drugcombdb.denglab.org/download/drug_protein_links.rar',
            'http://drugcombdb.denglab.org/download/protein_protein_links.rar',
            'http://drugcombdb.denglab.org/download/drug_chemical_info.csv'
        ]

        for url in urls:
            download_url(root_dir, url)

        patoolib.extract_archive(self.raw_dir + 'drug_protein_links.rar', self.raw_dir)
        patoolib.extract_archive(self.raw_dir + 'protein_protein_links.rar', self.raw_dir)

    def process(self):
        node_ftrs = self._get_node_ftrs()
        edge_idxs, edge_attr, graph_type_idx_ranges = self._get_edges_information()

        data = Data(x=torch.tensor(node_ftrs, dtype=torch.float),
                        edge_index=torch.tensor(edge_idxs, dtype=torch.long),
                        edge_attr=edge_attr)

        data.graph_type_idx_ranges = graph_type_idx_ranges

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_node_ftrs(self):
        nodes = self._get_nodes()
        return nodes.to_numpy().astype(np.int)

    def _get_nodes(self):
        drug_nodes = self._get_drug_nodes()
        protein_nodes = self._get_protein_nodes()

        nodes = pd.concat((drug_chem_info_with_fp, protein_nodes), ignore_index=True, sort=False)
        nodes = nodes.fillna(-1)

        return nodes

    def _get_drug_nodes(self):
        drug_info_filename = self.raw_dir + self.raw_file_names[0]
        drug_chem_info_no_fp = pd.read_csv(drug_info_filename)
        drug_chem_info_with_fp = self._augment_drug_info_with_fp(drug_chem_info_no_fp)

        drug_chem_info_with_fp['has_fp'] = drug_chem_info_with_fp['fp0'].apply(lambda fp: fp != -1)
        drug_chem_info_with_fp['is_drug'] = 1
        drug_chem_info_with_fp = drug_chem_info_with_fp.rename(columns={'drugName': 'name'})

        drug_nodes = drug_chem_info_with_fp.drop(
            ['cIds', 'drugNameOfficial', 'molecularWeight', 'smilesString', 'name'], axis=1)

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
        all_fp = pd.DataFrame(all_fp, columns=["fp" + str(i) for i in range(nBits)])

        return pd.concat((drug_chem_info_no_fp, all_fp), axis=1)

    def _get_protein_nodes(self):
        all_proteins = set(self.drug_protein_link['protein']).union(
            set(self.protein_protein_interactions['protein1'])
        ).union(set(self.protein_protein_interactions['protein2']))

        protein_nodes = pd.DataFrame(all_proteins, columns=['name'])
        protein_nodes['is_drug'] = 0
        protein_nodes['has_fp'] = False
        protein_nodes = protein_nodes.drop(
            ['cIds', 'drugNameOfficial', 'molecularWeight', 'smilesString', 'name'], axis=1)

        return protein_nodes

    def _get_edges_information(self):
        ppi_edge_idx, ppi_edge_attr = self._get_ppi_edges()
        dpi_edge_idx, dpi_edge_attr = self._get_dpi_edges()
        ddi_edge_idx, ddi_edge_attr = self._get_ddi_edges()

        all_edge_idxs = np.concatenate((ppi_edge_idx, dpi_edge_idx, ddi_edge_idx), axis=1)
        all_edge_attr = np.concatenate((ppi_edge_attr, dpi_edge_attr, ddi_edge_attr), axis=0)

        # Edges are directed, we need to feed them both ways
        all_edge_idxs = np.concatenate((all_edge_idxs, all_edge_idxs[::-1, :]), axis=1)
        all_edge_attr = np.concatenate((all_edge_attr, all_edge_attr), axis=0)

        graph_type_idx_ranges = self._get_graph_type_idx_ranges(ppi_edge_idx, dpi_edge_idx, ddi_edge_idx)

        return all_edge_idxs, all_edge_attr, graph_type_idx_ranges

    def _get_ddi_edges(self):
        drug_scores_filename = self.raw_dir + self.raw_file_names[1]
        drug_comb_scored = pd.read_csv(drug_scores_filename)

        drug_comb_scored['idx_Drug1'] = drug_comb_scored['Drug1'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)
        drug_comb_scored['idx_Drug2'] = drug_comb_scored['Drug2'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        # Remove measures that have been performed several times.
        drug_drug_edges = drug_comb_scored[drug_comb_scored[['idx_Drug1', 'idx_Drug2']].duplicated() == False]
        # Remove measures for which there is no information about one of the drugs
        drug_drug_edges = drug_drug_edges[drug_drug_edges['idx_Drug1'] != -1]
        drug_drug_edges = drug_drug_edges[drug_drug_edges['idx_Drug2'] != -1]

        ddi_edge_idxs = drug_drug_edges[['idx_Drug1', 'idx_Drug2']].to_numpy().T
        ddi_edge_attr = drug_drug_edges[['ZIP', 'Bliss', 'Loewe', 'HSA']].to_numpy()

        return ddi_edge_idxs, ddi_edge_attr

    def _get_dpi_edge_indices(self):
        self.drug_protein_link['idx_chemical'] = self.drug_protein_link['chemical'].apply(
            lambda s: cid_to_idx_dict[s] if s in cid_to_idx_dict.keys() else -1)

        self.drug_protein_link['idx_prot'] = \
            self.drug_protein_link['protein'].apply(lambda s: name_to_idx_dict[s])

        # Remove items without chem info
        links_all_chem_info = self.drug_protein_link[self.drug_protein_link['idx_chemical'] != -1]

        dpi_edge_idxs = links_all_chem_info[['idx_chemical', 'idx_prot']].to_numpy().T
        dpi_edge_attr = np.zeros((ppi_edge_index.shape[1], 4))

        return dpi_edge_idxs, dpi_edge_attr

    def _get_ppi_edge_indices(self):
        self.protein_protein_interactions['idx_prot1'] = \
            self.protein_protein_interactions['protein1'].apply(lambda s: name_to_idx_dict[s])

        self.protein_protein_interactions['idx_prot2'] = \
            self.protein_protein_interactions['protein2'].apply(lambda s: name_to_idx_dict[s])

        ppi_edge_idxs = self.protein_protein_interactions[['idx_prot1', 'idx_prot2']].to_numpy().T
        ppi_edge_attr = np.zeros(ppii_edge_idxs.shape[1], 4)

        return ppi_edge_idxs, ppi_edge_attr

    def _get_graph_type_idx_ranges(self, ppi_edge_idx, dpi_edge_idx, ddi_edge_idx):
        graph_type_idx_ranges = {}

        graph_type_idx_ranges['ppi'] = (0, ppi_edge_idx.shape[0])
        graph_type_idx_ranges['dpi'] = (
            graph_type_idx_ranges['ppi'][1],
            graph_type_idx_ranges['ppi'][1] + dpi_edge_idx.shape[0]
        )
        graph_type_idx_ranges['ddi']  = (
            graph_type_idx_ranges['dpi'][1],
            graph_type_idx_ranges['dpi'][1] + ddi_edge_idx.shape[0]
        )

        return graph_type_idx_ranges

if __name__ == '__main__':
    dataset = DrugCombDb(pre_transform=transform_to_bipartite_drug_protein_graph)

