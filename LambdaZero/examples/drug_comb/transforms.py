from torch_geometric.data import Data
import numpy as np
import torch

def to_drug_induced_subgraphs(data_list):
    """Restructures graphs to their ddis and protein subgraphs for each drug.

    Refer to the docs for _build_subgraphs_data for detailed information
    about this method.

    Should be used as a pre_transform method.
    """
    return [_build_subgraphs_data(data) for data in data_list]

def _build_subgraphs_data(data):
    """Restructures a graph to its ddi and protein subgraphs for each drug.

    Given a graph object, this method creates a new graph object whose
    edge index is restricted only to the drug-drug edge index of the
    input graph, while removing any edges which contain a drug which
    has no protein targets.

    The returned graph object also contains another attribute,
    drug_idx_to_graph, a dictionary wherein the keys are drugs
    and the values are Data objects which are the protein subgraphs
    for these drugs.  The protein subgraphs are defined as described
    in the docs for the _build_subgraph method.

    Parameters
    ----------
    data : Data
        The original "giant" graph object.  This graph is the undirected
        union of the drug-drug, drug-protein, and protein-protein graphs.

    Returns
    -------
    Data
        A modified version of the input graph object.  This graph's nodes
        are exclusively drug nodes and edges are only drug-drug edges where
        each drug in the edge has at least one protein target.  The Data
        object has three additional attributes besides the typical attributes
        of a Data object.  These are the following:

        protein_ftrs : np.ndarray
            An ndarray of shape (num_protein_nodes, num_features) holding the
            features for all protein nodes.
        edge_classes : np.ndarray
            An ndarray of shape (num_ddi_edges,) which records the cell line
            for each drug-drug edge.  The cell line is recorded as a simple
            integer value encoding.  E.g., the first cell line is a 0, the
            second is a 1, the third a 2, and so on.
        drug_idx_to_graph : Dict[int, Data]
            The keys in this dictionary are drugs, while the values are the
            subgraphs for these drugs.  The protein subgraphs are built as
            described in the docs for the _build_subgraph method.

    """
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
    drug_graphs = [_build_subgraph(data, drug_idx_to_split, drug, len(unique_drugs)) for drug in unique_drugs]

    drug_idx_to_graph = {unique_drugs[i]: drug_graph for i, drug_graph in enumerate(drug_graphs)}
    drug_drug_index = data.edge_index[:, ddi_edge_idx_range[0]:ddi_edge_idx_range[1]]
    drug_indexer = _get_drug_indexer(drug_drug_index, drug_idx_to_graph)

    # Create one graph object whose edges are simply the ddi edges, but augment
    # the graph object with the attribute drug_idx_to_graph which maps a drug
    # to its subgraph which can subsequently be used to compute a subgraph embedding.
    super_graph = Data(
        x=data.x[:len(unique_drugs)],
        edge_index=drug_drug_index[:, drug_indexer],
        y=data.y[ddi_edge_idx_range[0]:ddi_edge_idx_range[1]][drug_indexer],
    )

    super_graph.protein_ftrs = data.x[len(unique_drugs):]
    super_graph.edge_classes = data.edge_classes[ddi_edge_idx_range[0]:ddi_edge_idx_range[1]][drug_indexer]
    super_graph.drug_idx_to_graph = drug_idx_to_graph

    return super_graph

def _build_subgraph(parent_graph, drug_idx_to_split, drug, num_drug_nodes):
    """Builds the protein subgraph for a drug given that drug.

    Say we have three graphs,
    ..math::
    G_{ddi} = (V_{ddi}, E_{ddi}) \\
    G_{dpi} = (V_{dpi}, E_{dpi}) \\
    G_{ppi} = (V_{ppi}, E_{ppi})

    Then, we have our giant graph :math:`G = (V, E)` where
    ..math::
    V = V_{ddi} \cup V_{dpi} \cup V_{ppi} \\
    E = E_{ddi} \cup E_{dpi} \cup V_{ppi}

    Given a drug :math:`d \in V_{ddi}`, this method constructs a subgraph
    :math:`G_d = (V_d, E_d)` where,
    ..math::
    V_d = {p | p \in V_{ppi} and (d, p) \in E_{dpi}}
    E_d = {(p, p') | p, p' \in V_d and (p, p') \in E_{ppi}}

    In words, this method returns, for some drug d, the subgraph wherein
    the subgraph's nodes are all of the targets for that drug and the edges
    are edges in the ppi between nodes which are targets of drug d.

    Parameters
    ----------
    parent_graph : Data
        The "giant" graph G described above.
    drug_idx_to_split : Dict[int, np.ndarray]
        The keys in this dictionary are drugs, while the values are the
        edge indexes of shape (2, num_protein_targets_of_drug) detailing
        the edges in :math:`M_d = {(d, p) | (d, p) \in E_{dpi}}`.
    drug : int
        The drug :math:`d` to build the subgraph for.
    num_drug_nodes : int
        The number of drug nodes, :math:`|V_{ddi}|`.
    """
    if drug not in drug_idx_to_split:
        return Data(edge_index=torch.tensor([], dtype=torch.long))

    drug_subgraph_idx = drug_idx_to_split[drug]
    nodes_in_subgraph = np.unique(drug_subgraph_idx)

    n_mask = np.zeros(parent_graph.num_nodes, dtype=np.bool_)
    n_mask[nodes_in_subgraph] = 1

    mask = n_mask[parent_graph.edge_index[0]] & n_mask[parent_graph.edge_index[1]]
    subgraph_edge_index = parent_graph.edge_index[:, mask]

    # remove drug node from the subgraph here.  edges_without_drug is a bool array
    # wherein an item is True if the edge does not contain the drug node, and
    # False if it does contain the drug edge
    edges_without_drug = ~(subgraph_edge_index.T == drug).any(-1)
    subgraph_edge_index = subgraph_edge_index[:,edges_without_drug]

    # Make edge index relative to a 0-indexed protein feature matrix
    subgraph_edge_index -= num_drug_nodes

    # Maintain the protein nodes here as well
    subgraph = Data(edge_index=torch.tensor(subgraph_edge_index, dtype=torch.long))
    subgraph.nodes = torch.unique(subgraph.edge_index.flatten())

    return subgraph

def _get_drug_indexer(drug_drug_index, drug_idx_to_graph):
    """Builds an array indexer of valid ddi edges.

    Parameters
    ----------
    drug_drug_index : np.ndarray
        An ndarray for the drug drug edge index.  This is of shape
        (2, num_ddi_edges).
    drug_idx_to_graph : Dict[int, Data]
        The keys in this dictionary are drugs, while the values are the
        subgraphs for these drugs.  The protein subgraphs are built as
        described in the docs for the _build_subgraph method.

    Returns
    -------
    np.ndarray[bool]
        A boolean ndarray of shape (num_ddi_edges,) wherein the ith
        item of the ndarray is True if the ith edge of the
        drug_drug_index :math:`(d, d')` is not a self loop and if
        both drugs d and d' both have at least one protein target.
        The ith component is False if these conditions are not met.
    """
    drugs_without_target = torch.tensor([
        drug for drug, graph in drug_idx_to_graph.items()
        if len(graph.edge_index) == 0 or graph.edge_index.shape[1] == 0
    ])

    # Tried to figure out how to do this vectorized, the below _almost_ works
    #
    # (drug_drug_index[:, None].T == drugs_without_targets.T).all(-1).any(-1)
    #
    # But it only finds self loops it seems.  There's only 65k drug edges so a one-time
    # loop isn't too bad.  Just doing this to save dev time...
    bad_drug_indices = torch.zeros(drug_drug_index.shape[1]).to(torch.bool)
    for i in range(drug_drug_index.shape[1]):
        if drug_drug_index[0, i] in drugs_without_target or drug_drug_index[1, i] in drugs_without_target:
            bad_drug_indices[i] = True

    self_loop_indices = drug_drug_index[0] == drug_drug_index[1]

    return ~torch.stack((self_loop_indices, bad_drug_indices)).any(0)

def to_bipartite_drug_protein_graph(data_list):
    """Alters each Data object in the data_list to be only the bipartite graph.

    This should be used as a pre_transform.

    Parameters
    ----------
    data_list : List[Data]
        A list of Data object to alter.

    Returns
    -------
    List[Data]
        The original data objects but with only the drug-protein bipartite graph.
    """
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

def subgraph_protein_features_to_embedding(embedding_size, device):
    """Converts protein features from a one-hot encoding to a learnable tensor.

    Parameters
    ----------
    embedding_size : int
        The size of the embedding.
    device : {'cpu', 'cuda'}
        The device to initialize the embedding on.
    data : Data
        The induced subgraph graph object.

    Returns
    -------
    Data
        The original data object but with protein_ftrs as a learnable tensor
        instead of a one-hot encoding.

    Notes
    -----
    These docs combine information for the closure and its enclosing method
    for brevity.
    """
    def _subgraph_protein_features_to_embedding(data):
        if not hasattr(data, 'drug_idx_to_graph'):
            raise RuntimeError(
                'Data object does not have an attribute drug_idx_to_graph. ' +
                'It must have this to use the transform subgraph_protein_features_to_embedding.'
            )

        data.protein_ftrs = torch.rand((data.protein_ftrs.shape[0], embedding_size),
                                       requires_grad=True, dtype=torch.float, device=device)

        return data

    return _subgraph_protein_features_to_embedding

def use_score_type_as_target(score_type):
    """Alters the data object to only have targets of one cell type.

    Parameters
    ----------
    score_type : {'zip', 'bliss', 'loewe', 'hsa'}
        The score type to use in the alterred graph.

    Returns
    -------
    Data
        The original data object but with only the requested score type in
        the Data's y attribute.

    Raises
    ------
    KeyError
        Raised if the parameter score_type is not one of the four
        allowed values.

    Notes
    -----
    These docs combine information for the closure and its enclosing method
    for brevity.
    """
    # Index in final tensors of targets for each score type
    score_type_to_idx = {
        'zip':   0,
        'bliss': 1,
        'loewe': 2,
        'hsa':   3,
    }

    score_type = score_type.lower()
    score_idx = score_type_to_idx[score_type]

    def _use_score_type_as_target(data):
        data.y = data.y[:, score_idx]
        return data

    return _use_score_type_as_target

def train_first_k_edges(k):
    """Alters the data object to only have k edges.

    Parameters
    ----------
    k : int
        The number of edges to have in the graph.

    Returns
    -------
    Data
        The original data object but with only the requested number of
        edges in the Data object.

    Notes
    -----
    These docs combine information for the closure and its enclosing method
    for brevity.
    """
    def _train_first_k_edges(data):
        data.edge_index = data.edge_index[:, :k]
        data.edge_classes = data.edge_classes[:k]
        data.y = data.y[:k]

        return data

    return _train_first_k_edges

def use_single_cell_line(data):
    """Alters a data object to only have edges for a single cell line.

    Parameters
    ----------
    data : Data
        The data object to be altered.

    Returns
    -------
    Data
        The original data object, but only containing edges which belong to
        experiments on the most populous cell line.
    """
    cell_lines, cell_line_counts = torch.unique(data.edge_classes, return_counts=True)
    cell_line = cell_lines[torch.argmax(cell_line_counts)]

    # nonzero returns a tensor of size n x 1, but we want 1 x n, so flatten
    matching_indices = (data.edge_classes == cell_line).nonzero().flatten()

    data.edge_index = data.edge_index[:, matching_indices]
    data.y = data.y[matching_indices]
    data.edge_classes = data.edge_classes[matching_indices]

    return data

