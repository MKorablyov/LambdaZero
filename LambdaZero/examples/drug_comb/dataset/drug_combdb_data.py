from LambdaZero.utils import get_external_dirs
from torch_geometric.data import Data, InMemoryDataset, download_url
from rdkit import Chem
from rdkit.Chem import AllChem
# from unrar import rarfile
import numpy as np
import pandas as pd
import torch

def get_fingerprint(smile, radius, n_bits):
    if smile == 'none':
        return np.array([-1]*n_bits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
    except Exception as ex:
        return np.array([-1]*n_bits)

def to_drug_induced_subgraphs(data_list):
    new_data_list = []
    for data in data_list:
        all_edge_idx = data.edge_index

        dpi_edge_idx_range = data.graph_type_idx_ranges['dpi']
        dpi_edge_idx = all_edge_idx[:, dpi_edge_idx_range[0]:dpi_edge_idx_range[1]]

        # Get the edges in order so that edges for the same drug are all
        # grouped next to each other
        idx_sorts = dpi_edge_idx[0,:].argsort()
        re_index = dpi_edge_idx[:, idx_sorts]

        # np.unique returns the index that each unique drug (i.e., value of re_index[0,:])
        # occurs at. Since the edges were ordered with argsort above, consecutive items of
        # idx_start represent begin and end ranges for a particular drug's edges.  Then,
        # use these begin and end ranges to split the original edge index into a separate
        # one for each drug with np.split
        all_idxs = np.arange(re_index.shape[1])
        _, idx_start = np.unique(re_index[0,:], return_index=True)
        drug_to_prot_sorted = [re_index[:, split] for split in np.split(all_idxs, idx_start[1:])]

        # split[0,0] is a tensor, so call item() on it to get the int out
        drug_idx_to_split = {split[0,0].item(): split for split in drug_to_prot_sorted}

        # Get all drugs by taking union of drugs in ddi_edge_idx and re_index
        # (that is, the dpi graph drugs)
        ddi_edge_idx_range = data.graph_type_idx_ranges['ddi']
        ddi_edge_idx = all_edge_idx[:, ddi_edge_idx_range[0]:ddi_edge_idx_range[1]]
        unique_drugs = np.unique(np.hstack([ddi_edge_idx.flatten(), re_index[0,:]]))

        # Create a new graph for each drug...
        drug_graphs = []
        for drug in unique_drugs:
            new_graph = None
            if drug not in drug_idx_to_split:
                new_graph = Data(x=data.x[drug].reshape(1, -1),
                                 edge_index=torch.tensor([], dtype=torch.long))
            else:
                drug_subgraph_idx = drug_idx_to_split[drug]
                protein_ftrs = data.x[np.sort(drug_subgraph_idx[1,:])]
                drug_ftr = data.x[drug_subgraph_idx[0,0]].reshape(1, -1)

                ftrs = torch.cat((drug_ftr, protein_ftrs), dim=0)

                # Convert edge index to format for small graph. If we did this,
                # the edge index for each drug graph would have the indices for the
                # original large graph which could create unnecessarily large adjacency
                # matrices later in execution.  Because of this, here we change the
                # indices for the small graph's edge index to start at 0 (using
                # np.unique and np.digitize).

                # Use np.unique + 1 to create bins for use in np.digitize
                bins = np.unique(drug_subgraph_idx[1,:]) + 1

                # Add 1 here so the 0th index of edges will refer to the drug node
                drug_subgraph_idx[1,:] = torch.from_numpy(np.digitize(drug_subgraph_idx[1,:], bins) + 1)
                drug_subgraph_idx[0,:] = 0

                new_graph = Data(x=ftrs, edge_index=drug_subgraph_idx)

            drug_graphs.append(new_graph)

        # Create one graph object whose edges are simply the ddi edges, but augment
        # the graph object with the attribute drug_idx_to_graph which maps a drug
        # to its subgraph which can subsequently be used to compute a subgraph embedding.
        super_graph = Data(
            x=data.x[:len(unique_drugs)],
            edge_index=data.edge_index[:, ddi_edge_idx_range[0]:ddi_edge_idx_range[1]],
            y=data.y[ddi_edge_idx_range[0]:ddi_edge_idx_range[1]],
        )

        super_graph.edge_classes = data.edge_classes[ddi_edge_idx_range[0]:ddi_edge_idx_range[1]]
        super_graph.drug_idx_to_graph = {unique_drugs[i]: drug_graph for i, drug_graph in enumerate(drug_graphs)}

        new_data_list.append(super_graph)

    return new_data_list


def to_bipartite_drug_protein_graph(data_list):
    import pdb; pdb.set_trace()
    new_data_list = []
    for data in data_list:
        ddi_edge_idx_range = data.graph_type_idx_ranges['ddi']
        dpi_edge_idx_range = data.graph_type_idx_ranges['dpi']

        pdi_offset = ddi_edge_idx_range[1]
        pdi_edge_idx_range = [idx + pdi_offset for idx in dpi_edge_idx_range]

        new_edge_idx_first = data.edge_index[:, dpi_edge_idx_range[0]:dpi_edge_idx_range[1]]
        new_edge_idx_scnd = data.edge_index[:, pdi_edge_idx_range[0]:pdi_edge_idx_range[1]]

        new_edge_idx = torch.cat((new_edge_idx_first, new_edge_idx_scnd), dim=1)

        ddi_edges = data.edge_index[:, ddi_edge_idx_range[0]:ddi_edge_idx_range[1]]
        super_graph = Data(
            x=data.x[:len(torch.unique(ddi_edges.flatten()))],
            edge_index=ddi_edges,
            y=data.y[ddi_edge_idx_range[0]:ddi_edge_idx_range[1]],
        )

        super_graph.edge_classes = data.edge_classes[ddi_edge_idx_range[0]:ddi_edge_idx_range[1]]
        super_graph.drug_protein_graph = Data(
            x=data.x,
            edge_index=torch.tensor(new_edge_idx, dtype=torch.long),
        )

        new_data_list.append(super_graph)

    return new_data_list


class DrugCombDb(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None, fp_bits=1024, fp_radius=4,
                 scores=['ZIP', 'Bliss', 'Loewe', 'HSA']):
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius
        
        self._drug_protein_link_holder = None
        self._protein_protein_interactions_holder = None

        self.scores = scores
        
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
        nodes, number_of_drugs = self._get_nodes()
        edge_idxs, edge_trgt, edge_classes, graph_type_idx_ranges = self._get_edges_information(nodes)

        data = Data(
            x=torch.tensor(self._get_node_ftrs(nodes), dtype=torch.float),
            edge_index=torch.tensor(edge_idxs, dtype=torch.long),
            y=torch.tensor(edge_trgt, dtype=torch.float),
        )

        data.edge_classes = torch.tensor(edge_classes, dtype=torch.long)
        data.graph_type_idx_ranges = graph_type_idx_ranges
        data.number_of_drugs = number_of_drugs

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_node_ftrs(self, nodes):
        node_ftrs = nodes.fillna(-1)
        node_ftrs = node_ftrs.drop(['cIds', 'drugNameOfficial', 'molecularWeight',
                                    'smilesString', 'name', 'has_fp'], axis=1)

        # Move 'is_drug' column to front
        is_drug = node_ftrs['is_drug']
        node_ftrs.drop(labels=['is_drug'], axis=1, inplace=True)
        node_ftrs.insert(0, 'is_drug', is_drug)

        return node_ftrs.to_numpy().astype(np.int)

    def _get_nodes(self):
        drug_nodes = self._get_drug_nodes()
        protein_nodes = self._get_protein_nodes()

        nodes = pd.concat((drug_nodes, protein_nodes), ignore_index=True, sort=False)

        number_of_drugs = drug_nodes.shape[0]

        return nodes, number_of_drugs

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
        all_fp = drug_chem_info_no_fp['smilesString'].apply(
            lambda s: get_fingerprint(s, radius=self.fp_radius, n_bits=self.fp_bits)
        )

        # Convert to dataframe
        all_fp = list(all_fp)
        all_fp = [list(fp) for fp in all_fp]
        all_fp = pd.DataFrame(all_fp, columns=["fp" + str(i) for i in range(self.fp_bits)])

        return pd.concat((drug_chem_info_no_fp, all_fp), axis=1)

    def _get_protein_nodes(self):
        print('Processing protein nodes..')

        all_proteins = set(self._drug_protein_link['protein']).union(
            set(self._protein_protein_interactions['protein1'])
        ).union(set(self._protein_protein_interactions['protein2']))

        protein_nodes = pd.DataFrame(all_proteins, columns=['name'])
        protein_nodes['is_drug'] = 0
        protein_nodes['has_fp'] = False

        one_hot = pd.get_dummies(protein_nodes['name'])
        protein_nodes = protein_nodes.join(one_hot)

        return protein_nodes

    def _get_edges_information(self, nodes):
        print('Processing edges..')

        cid_to_idx_dict = {nodes.at[i, 'cIds']: i for i in range(len(nodes))}
        name_to_idx_dict = {nodes.at[i, 'name']: i for i in range(len(nodes))}

        ppi_edge_idx, ppi_edge_trgt, ppi_edge_classes = self._get_ppi_edges(name_to_idx_dict)
        dpi_edge_idx, dpi_edge_trgt, dpi_edge_classes = self._get_dpi_edges(cid_to_idx_dict, name_to_idx_dict)
        ddi_edge_idx, ddi_edge_trgt, ddi_edge_classes = self._get_ddi_edges(name_to_idx_dict)

        all_edge_idxs = np.concatenate((ppi_edge_idx, dpi_edge_idx, ddi_edge_idx), axis=1)
        all_edge_trgt = np.concatenate((ppi_edge_trgt, dpi_edge_trgt, ddi_edge_trgt), axis=0)
        all_edge_classes = np.concatenate((ppi_edge_classes, dpi_edge_classes, ddi_edge_classes))

        # Edges are directed, we need to feed them both ways
        all_edge_idxs = np.concatenate((all_edge_idxs, all_edge_idxs[::-1, :]), axis=1)
        all_edge_trgt = np.concatenate((all_edge_trgt, all_edge_trgt), axis=0)
        all_edge_classes = np.concatenate((all_edge_classes, all_edge_classes))

        graph_type_idx_ranges = self._get_graph_type_idx_ranges(ppi_edge_idx, dpi_edge_idx, ddi_edge_idx)

        return all_edge_idxs, all_edge_trgt, all_edge_classes, graph_type_idx_ranges

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

        # Keep the first score for each (drug, drug, cell line) tuple that exists
        drug_drug_edges = drug_drug_edges.drop_duplicates(['idx_Drug1', 'idx_Drug2', 'Cell line'])

        # Remove edges with invalid scores
        scores = drug_drug_edges[self.scores]
        bad_scores_arr = ((scores.notna()) & (-100 <= scores) & (scores <= 100)).all(axis=1)
        drug_drug_edges = drug_drug_edges[bad_scores_arr]

        cell_lines = drug_drug_edges['Cell line'].unique()
        mapping = {cell_line: i for i, cell_line in enumerate(cell_lines)}
        ddi_edge_classes = drug_drug_edges.replace({'Cell line': mapping})['Cell line']
        ddi_edge_classes = ddi_edge_classes.to_numpy()

        ddi_edge_idxs = drug_drug_edges[['idx_Drug1', 'idx_Drug2']].to_numpy().T
        ddi_edge_attr = drug_drug_edges[self.scores].to_numpy()

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

    dataset = DrugCombDb(pre_transform=to_bipartite_drug_protein_graph)

    dataset.get_edge_type('ddi')

