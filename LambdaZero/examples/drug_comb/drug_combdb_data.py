from LambdaZero.utils import get_external_dirs
from LambdaZero.examples.drug_comb.transforms import to_drug_induced_subgraphs
from torch_geometric.data import Data, InMemoryDataset, download_url
from rdkit import Chem
from rdkit.Chem import AllChem
from unrar import rarfile
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
import torch
import os

def get_fingerprint(smile, radius, n_bits):
    """Gets the morgan fingerprint for a SMILES string

    Parameters
    ----------
    smile : str
        The SMILES string of the object we wish to get a fingerprint for.
    radius : int
        The radius of the fingerprint
    n_bits : int
        The number of bits to store the fingerprint's representation in

    Returns
    -------
    np.ndarray
        An np.array of shape (n_bits,) containing either the fingerprint,
        or all -1s if we could not obtain the fingerprint.

    """
    if smile == 'none':
        return np.array([-1]*n_bits)
    try:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), radius, n_bits))
    except Exception as ex:
        return np.array([-1]*n_bits)

class DrugCombDb(InMemoryDataset):
    """A dataset class holding data corresponding to the DrugCombDb dataset.

    A dataset object compatible with pytorch-geometric and torch infrastructure.
    This dataset object holds information pertaining to the DrugCombDb dataset.
    The dataset represents a link prediction task, wherein we would like to
    predict synergy scores for pairs of drugs in the dataset for specific
    cell lines.

    Notes
    -----
        If left unaltered from transform or pre_transform methods, this dataset
        contains a graph which is the union of a drug-drug interaction graph,
        a drug-protein target graph, and a protein-protein interaction graph.
        The graph is undirected, such that if some edge `A -> B` exists in
        the dataset's edge_index, then the edge `B -> A` also exists in the
        edge_index.

    Attributes
    ----------
    fp_bits : int
        For all drugs in the dataset we obtain a morgan fingerprint.  This is the
        number of bits used for said fingerprints.
    fp_radius : int
        For all drugs in the dataset we obtain a morgan fingerprint.  This is the
        radius for said fingerprints.
    use_laplacian : bool
        If True, use laplacian embeddings as protein features.  If False, use one-hot
        encodings.
    target_type : {'response', 'synergy_scores'}
        If 'response', use the raw drug response as the target.  If 'score', use the
        synergy scores as the target.
    _drug_protein_link_holder : pd.DataFrame
        A convenience DataFrame maintaining the drug-protein targets for processing
        in different parts of the code so that the DataFrame does not need to be
        read from disk multiple times.
    _protein_protein_interactions_holder : pd.DataFrame
        A convenience DataFrame maintaining the protein-protein interactions for processing
        in different parts of the code so that the DataFrame does not need to be
        read from disk multiple times.
    """

    def __init__(self, transform=None, pre_transform=None, prot_ftr_type='one_hot',
                 target_type='response', fp_bits=1024, fp_radius=4):
        self._get_prot_ftrs = self._get_get_prot_ftr_method(prot_ftr_type)
        self._get_ddi_edges = self._get_get_ddi_edges_method(target_type)
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius

        self.edge_attr_size = 2 if target_type == 'response' else 0
        self.edge_trgt_size = 1 if target_type == 'response' else 4

        self._drug_protein_link_holder = None
        self._protein_protein_interactions_holder = None

        datasets_dir, _, _ = get_external_dirs()
        super().__init__(datasets_dir + '/DrugCombDb/', transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    def _get_get_prot_ftr_method(self, prot_ftr_type):
        """Gets the method which builds the protein features

        Arguments
        ---------
        prot_ftr_type : {'laplacian', 'one_hot'}
            String with value of either laplacian or one_hot.

        Returns
        -------
        Function[pd.DataFrame, pd.DataFrame]
            Method which will compute protein features
        """
        if prot_ftr_type not in {'laplacian', 'one_hot'}:
            raise ValueError(
                "prot_ftr_type must be one of 'laplacian' or 'one_hot'.  " +
                "Was %s instead" % prot_ftr_type
            )

        method_dict = {'laplacian': self._get_laplacian_features,
                       'one_hot': self._get_one_hot_features}

        return method_dict[prot_ftr_type]

    def _get_get_ddi_edges_method(self, target_type):
        """Gets the method which builds the targets

        Arguments
        ---------
        target_type : {'response', 'synergy_scores'}
            String with value of either response or synergy_scores.

        Returns
        -------
        Function[pd.DataFrame, pd.DataFrame]
            Method which will compute drug pair targets
        """
        if target_type not in {'response', 'synergy_scores'}:
            raise ValueError(
                "prot_ftr_type must be one of 'laplacian' or 'one_hot'.  " +
                "Was %s instead" % target_type
            )

        method_dict = {'response': self._get_response_ddi_edges,
                       'synergy_scores': self._get_synergy_score_ddi_edges}

        return method_dict[target_type]

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
            'protein_protein_links.txt',
            'drugcombs_response.csv',
        ]

    @property
    def processed_file_names(self):
        return ['drug_combdb_data.pt']

    def download(self):
        urls = [
            'http://drugcombdb.denglab.org/download/drugcombs_scored.csv',
            'http://drugcombdb.denglab.org/download/drugcombs_response.csv',
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
        """The main dataset processing method according to torch dataset convention.

        This method, in order, gets nodes, gets edges, creates the graph (the graph
        is represented by the Data object), and applies the pre_transform method if
        one exists to this graph.  The method also adds the attributes edge_classes
        and graph_type_idx_ranges to the graph so that they may be accessed at a later
        time.  Refer to the documentation for _get_edges_information for details on
        these objects.

        Returns
        _______
        None

        """
        nodes = self._get_nodes()
        edge_idxs, edge_trgt, edge_attr, edge_classes, graph_type_idx_ranges = self._get_edges_information(nodes)

        data = Data(
            x=torch.tensor(self._get_node_ftrs(nodes), dtype=torch.float),
            edge_index=torch.tensor(edge_idxs, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            y=torch.tensor(edge_trgt, dtype=torch.long),
        )

        data.edge_classes = torch.tensor(edge_classes, dtype=torch.long)
        data.graph_type_idx_ranges = graph_type_idx_ranges

        data_list = [data]
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _get_node_ftrs(self, nodes):
        """Gets the feature matrix for all graph nodes.

        Parameters
        ----------
        nodes : pd.DataFrame
            Each row in the nodes DataFrame is a node in the graph.

        Returns
        -------
        np.ndarray
            ndarray of the node features. The features are a concatenation
            of a morgan fingerprint for drug nodes and a one-hot encoding
            of proteins for the protein nodes.
        """
        node_ftrs = nodes.fillna(-1)
        node_ftrs = node_ftrs.drop(['cIds', 'drugNameOfficial', 'molecularWeight',
                                    'smilesString', 'name', 'has_fp', 'is_drug'], axis=1)

        return node_ftrs.to_numpy().astype(np.int)

    def _get_nodes(self):
        """Combines the drug and protein nodes and returns them.

        Returns
        -------
        pd.DataFrame
            Protein and drug nodes.
        """
        drug_nodes = self._get_drug_nodes()
        protein_nodes = self._get_protein_nodes()

        nodes = pd.concat((drug_nodes, protein_nodes), ignore_index=True, sort=False)

        return nodes

    def _get_drug_nodes(self):
        """Processes drug nodes from the raw drug file.

        For each drug all we do is get a fingerprint and give a bit of additional
        information about this.

        Returns
        -------
        pd.DataFrame
            The drug nodes.  We add columns 'has_fp' and 'is_drug' to the nodes.
        """
        print('Processing drug nodes..')

        drug_info_filename = '%s/%s' % (self.raw_dir, self.raw_file_names[0])
        drug_chem_info_no_fp = pd.read_csv(drug_info_filename, encoding='ISO-8859-1')
        drug_chem_info_with_fp = self._augment_drug_info_with_fp(drug_chem_info_no_fp)

        # drop drugs without fingerprint
        drug_chem_info_with_fp['has_fp'] = drug_chem_info_with_fp['fp0'].apply(lambda fp: fp != -1)
        drug_chem_info_with_fp = drug_chem_info_with_fp[drug_chem_info_with_fp.has_fp != -1]

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
        """Processes the protein nodes from the raw protein file.

        To process we get protein nodes by taking the union of proteins
        in the PPI and drug-protein target graphs.  Then, we just add
        columns 'is_drug' ans 'has_fp'.

        Returns
        -------
        pd.DataFrame
            Each row is a protein node.
        """
        print('Processing protein nodes..')

        all_proteins = set(self._drug_protein_link['protein']).union(
            set(self._protein_protein_interactions['protein1'])
        ).union(set(self._protein_protein_interactions['protein2']))

        protein_nodes = pd.DataFrame(all_proteins, columns=['name'])
        protein_nodes['is_drug'] = 0
        protein_nodes['has_fp'] = False

        ftrs = self._get_prot_ftrs(protein_nodes)
        protein_nodes = protein_nodes.join(ftrs)

        return protein_nodes

    def _get_one_hot_features(self, protein_nodes):
        return pd.get_dummies(protein_nodes['name'])

    def _get_laplacian_features(self, protein_nodes, n_features=256):
        if not os.path.isfile(os.path.join(self.raw_dir, "laplacian_eigvects.npz")):
            ppi_edge_idx = self._extract_intermediate_ppi_edge_idx(protein_nodes)

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

            ## Restrict to protein nodes
            #L = L[self.is_drug.sum():, self.is_drug.sum():]

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
        col_names = ["lap_feat_" + str(i) for i in range(256)]

        return pd.DataFrame(eigvects[:, :n_features], columns=col_names)

    def _extract_intermediate_ppi_edge_idx(self, protein_nodes):
        name_to_idx = {protein_nodes.at[i, 'name']: i for i in range(len(protein_nodes))}

        intermed_ppi = self._protein_protein_interactions.copy()
        intermed_ppi['idx_prot1'] = intermed_ppi['protein1'].apply(lambda s: name_to_idx[s])
        intermed_ppi['idx_prot2'] = intermed_ppi['protein2'].apply(lambda s: name_to_idx[s])

        ppi_edge_idx = intermed_ppi[['idx_prot1', 'idx_prot2']].to_numpy().T
        ppi_edge_idx = np.concatenate((ppi_edge_idx, ppi_edge_idx[::-1, :]), axis=1)

        return ppi_edge_idx

    def _get_edges_information(self, nodes):
        """Gets the edges in the graph

        Get all the edges, then add the reversed edges such that if edge
        (A, B) is in the graph then (B, A) is also in the graph.

        Parameters
        ----------
        nodes : pd.DataFrame
            A DataFrame containing drug and protein nodes.

        Returns
        -------
        all_edge_idxs : np.ndarray
            The edge index for the entire graph.  This is the union of edges
            in the drug-drug interaction graph, the drug-protein interaction
            graph, and the protein-protein interaction graph.  It also contains
            the reversed edges as detailed in the method description.
        all_edge_trgt : np.ndarray
            An ndarray of shape (num_edges, 4).  For each edge, this ndarray
            contains values for the four synergy scores observed for the
            interaction between these two edges.  These scores are only
            meaningful for edges wherein the head and tail of the edge are both
            drug nodes.  For edges where the head or tail is not a drug, the
            edge's entry in this ndarray is a row of 0s.  The scores recorded
            are the ZIP, Bliss, Loewe, and HSA scores.  Note that the scores'
            indices in the ndarray are
                ZIP   : 0
                Bliss : 1
                Loewe : 2
                HSA   : 3
        all_edge_classes : np.ndarray
            An ndarray of shape (num_edges,).  Each entry in this ndarray
            represents the cell line for the edge's observation.  The cell
            line is recorded as a simple integer value encoding.  E.g.,
            the first cell line is a 0, the second is a 1, the third a 2, and so on.
            The values in this ndarray are only informative for drug-drug edges,
            and for all other types of edges the edge's value in this ndarray is
            a -1.
        graph_type_idx_ranges : Dict[str, Tuple[int]]
            A dictionary which represents what index ranges within the
            all_edge_idxs, all_edge_trgt, and all_edge_classes ndarrays
            represent which edge types.  Here, edge types are drug-drug
            edges, drug-protein edges, and protein-protein edges. The
            range tuples in the dict in the dict are inclusive at their
            head, exclusive at their tail.  The dict looks like:
            {
                'ddi': (ddi_range_start, ddi_range_end),
                'dpi': (dpi_range_start, dpi_range_end),
                'ppi': (ppi_range_start, ppi_range_end),
            }
        """
        print('Processing edges..')

        cid_to_idx_dict = {nodes.at[i, 'cIds']: i for i in range(len(nodes))}
        name_to_idx_dict = {nodes.at[i, 'name']: i for i in range(len(nodes))}

        ppi_edge_idx, ppi_edge_trgt, ppi_edge_attr, ppi_edge_classes = self._get_ppi_edges(name_to_idx_dict)
        dpi_edge_idx, dpi_edge_trgt, dpi_edge_attr, dpi_edge_classes = self._get_dpi_edges(cid_to_idx_dict,
                                                                                           name_to_idx_dict)
        ddi_edge_idx, ddi_edge_trgt, ddi_edge_attr, ddi_edge_classes = self._get_ddi_edges(name_to_idx_dict)

        import pdb; pdb.set_trace()
        all_edge_idxs = np.concatenate((ppi_edge_idx, dpi_edge_idx, ddi_edge_idx), axis=1)
        all_edge_trgt = np.concatenate((ppi_edge_trgt, dpi_edge_trgt, ddi_edge_trgt), axis=0)
        all_edge_attr = np.concatenate((ppi_edge_attr, dpi_edge_attr, ddi_edge_attr), axis=0)
        all_edge_classes = np.concatenate((ppi_edge_classes, dpi_edge_classes, ddi_edge_classes))

        # Edges are directed, we need to feed them both ways
        all_edge_idxs = np.concatenate((all_edge_idxs, all_edge_idxs[::-1, :]), axis=1)
        all_edge_trgt = np.concatenate((all_edge_trgt, all_edge_trgt), axis=0)
        all_edge_attr = np.concatenate((all_edge_attr, all_edge_attr), axis=0)
        all_edge_classes = np.concatenate((all_edge_classes, all_edge_classes))

        graph_type_idx_ranges = self._get_graph_type_idx_ranges(ppi_edge_idx, dpi_edge_idx, ddi_edge_idx)

        return all_edge_idxs, all_edge_trgt, all_edge_attr, all_edge_classes, graph_type_idx_ranges

    def _get_synergy_score_ddi_edges(self, name_to_idx_dict):
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

        # The DrugCombDb dataset was built by its authors through a scraping &
        # aggregation of a number of different drug combination datasets.  For
        # some drug pairs and cell lines, the authors found multiple results
        # across different sources where the synergy scores significantly
        # differed across trials.  For these volatile drug pairs & cell lines,
        # they include all of the different observations. In our case, since
        # it is not clear whether we should average these trials or do
        # something similar, we just keep the first instance of the drug pair
        # and cell line and drop the others.
        drug_drug_edges = drug_drug_edges.drop_duplicates(['idx_Drug1', 'idx_Drug2', 'Cell line'])

        # There are a number of drug pair/cell line combinatioins which have
        # scores far outside the -100 to 100 range.  We remove these so
        # that the data is more well behaved.  (One question, maybe we
        # shouldn't do this?  Perhaps including them and normalizing would
        # still work okay?  I'm not sure, could be worth trying.)
        scores = drug_drug_edges[['ZIP', 'Bliss', 'Loewe', 'HSA']]
        bad_scores_arr = ((scores.notna()) & (-100 <= scores) & (scores <= 100)).all(axis=1)
        drug_drug_edges = drug_drug_edges[bad_scores_arr]

        cell_lines = drug_drug_edges['Cell line'].unique()
        mapping = {cell_line: i for i, cell_line in enumerate(cell_lines)}
        ddi_edge_classes = drug_drug_edges.replace({'Cell line': mapping})['Cell line']
        ddi_edge_classes = ddi_edge_classes.to_numpy()

        ddi_edge_idxs = drug_drug_edges[['idx_Drug1', 'idx_Drug2']].to_numpy().T
        ddi_edge_trgt = drug_drug_edges[['ZIP', 'Bliss', 'Loewe', 'HSA']].to_numpy()
        ddi_edge_attr = np.zeros((ddi_edge_idxs.shape[1], 0))

        return ddi_edge_idxs, ddi_edge_trgt, ddi_edge_attr, ddi_edge_classes

    def _get_response_ddi_edges(self, name_to_idx_dict):
        print('Processing drug drug interaction edges..')

        drug_response_filename = '%s/%s' % (self.raw_dir, self.raw_file_names[4])
        drug_comb_response = pd.read_csv(drug_response_filename, encoding='ISO-8859-1')

        # Remove rows with nan values
        drug_comb_response = drug_comb_response.dropna(axis=0)

        drug_comb_response['idx_DrugRow'] = drug_comb_response['DrugRow'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        drug_comb_response['idx_DrugCol'] = drug_comb_response['DrugCol'].apply(
            lambda s: name_to_idx_dict[s] if s in name_to_idx_dict.keys() else -1)

        # Remove measures for which there is no information about one of the drugs
        drug_comb_response = drug_comb_response[drug_comb_response['idx_DrugRow'] != -1]
        drug_comb_response = drug_comb_response[drug_comb_response['idx_DrugCol'] != -1]

        # Remove combinations of a drug with itself
        is_pairs_of_diff_drugs = drug_comb_response[['idx_DrugRow', 'idx_DrugCol']].apply(
            lambda s: s['idx_DrugRow'] != s['idx_DrugCol'], axis=1)
        drug_comb_response = drug_comb_response[is_pairs_of_diff_drugs]

        ddi_edge_idx = drug_comb_response[['idx_DrugRow', 'idx_DrugCol']].to_numpy().T
        ddi_edge_classes = np.zeros((len(drug_comb_response)))  # For compatibility between datasets
        ddi_edge_trgt = drug_comb_response['Response'].to_numpy().reshape((-1, 1))
        ddi_edge_attr = drug_comb_response[['ConcRow', 'ConcCol']].to_numpy()

        # Log transform concentrations
        ddi_edge_attr[:, 0] = np.log(ddi_edge_attr[:, 0] + 1e-6)
        ddi_edge_attr[:, 1] = np.log(ddi_edge_attr[:, 1] + 1e-6)

        return ddi_edge_idx, ddi_edge_trgt, ddi_edge_attr, ddi_edge_classes

    def _get_dpi_edges(self, cid_to_idx_dict, name_to_idx_dict):
        print('Processing drug protein edges..')

        self._drug_protein_link['idx_chemical'] = self._drug_protein_link['chemical'].apply(
            lambda s: cid_to_idx_dict[s] if s in cid_to_idx_dict.keys() else -1)

        self._drug_protein_link['idx_prot'] = \
            self._drug_protein_link['protein'].apply(lambda s: name_to_idx_dict[s])

        # Remove items without chem info
        links_all_chem_info = self._drug_protein_link[self._drug_protein_link['idx_chemical'] != -1]

        dpi_edge_idxs = links_all_chem_info[['idx_chemical', 'idx_prot']].to_numpy().T
        dpi_edge_trgt = np.zeros((dpi_edge_idxs.shape[1], self.edge_trgt_size))
        dpi_edge_attr = np.zeros((dpi_edge_idxs.shape[1], self.edge_attr_size))
        dpi_edge_classes = np.full((dpi_edge_idxs.shape[1]), -1)

        return dpi_edge_idxs, dpi_edge_trgt, dpi_edge_attr, dpi_edge_classes

    def _get_ppi_edges(self, name_to_idx_dict):
        print('Processing protein protein edges..')

        self._protein_protein_interactions['idx_prot1'] = \
            self._protein_protein_interactions['protein1'].apply(lambda s: name_to_idx_dict[s])

        self._protein_protein_interactions['idx_prot2'] = \
            self._protein_protein_interactions['protein2'].apply(lambda s: name_to_idx_dict[s])

        ppi_edge_idxs = self._protein_protein_interactions[['idx_prot1', 'idx_prot2']].to_numpy().T
        ppi_edge_trgt = np.zeros((ppi_edge_idxs.shape[1], self.edge_trgt_size))
        ppi_edge_attr = np.zeros((ppi_edge_idxs.shape[1], self.edge_attr_size))
        ppi_edge_classes = np.full((ppi_edge_idxs.shape[1]), -1)

        return ppi_edge_idxs, ppi_edge_trgt, ppi_edge_attr, ppi_edge_classes

    def _get_graph_type_idx_ranges(self, ppi_edge_idx, dpi_edge_idx, ddi_edge_idx):
        """Builds a dictionary detailing which edges belong to the ddi, dpi, and ppi.

        Returns
        -------
        Dict[str, Tuple[int]]
            A dictionary which represents what index ranges within the
            all_edge_idxs, all_edge_trgt, and all_edge_classes ndarrays
            represent which edge types.  Here, edge types are drug-drug
            edges, drug-protein edges, and protein-protein edges. The
            range tuples in the dict in the dict are inclusive at their
            head, exclusive at their tail.  The dict looks like:
            {
                'ddi': (ddi_range_start, ddi_range_end),
                'dpi': (dpi_range_start, dpi_range_end),
                'ppi': (ppi_range_start, ppi_range_end),
            }
        """
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
        unique_ddi_edge_idx = self.data.edge_index.unique(dim=1)
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
    dataset = DrugCombDb(
        pre_transform=to_drug_induced_subgraphs,
        prot_ftr_type='laplacian',
        target_type='response'
    )

