import torch


def random_split(num_examples, test_prob, valid_prob):
    nvalid = int(num_examples * valid_prob)
    ntest = int(num_examples * test_prob)
    idx = torch.randperm(num_examples)

    train_idx = idx[ntest + nvalid:]
    val_idx = idx[:nvalid]
    test_idx = idx[:ntest]

    return train_idx, val_idx, test_idx


def get_ddi_edges(data):
    """
    Return ddi edges, classes and scores
    Only returns edges one way (not inverted ones)
    """
    # Retrieve edge indices, classes and scores
    ddi_edges = data.edge_index[:, data.graph_type_idx_ranges['ddi'][0]:data.graph_type_idx_ranges['ddi'][1]]
    ddi_edge_classes = data.edge_classes[data.graph_type_idx_ranges['ddi'][0]:data.graph_type_idx_ranges['ddi'][1]]
    ddi_y = data.y[data.graph_type_idx_ranges['ddi'][0]:data.graph_type_idx_ranges['ddi'][1]]

    return ddi_edges, ddi_edge_classes, ddi_y


def get_ppi_and_dpi_edges(data):
    """
    Returns ppi and dpi edges, both ways
    """

    ####################################################################################################################
    # ppi edges
    ####################################################################################################################

    # Retrieve edge indices
    ppi_edges = data.edge_index[:, data.graph_type_idx_ranges['ppi'][0]:data.graph_type_idx_ranges['ppi'][1]]

    # Same thing with reversed edges
    reversed_ppi_edges = data.edge_index[:, data.graph_type_idx_ranges['ppi'][0] + data.edge_index.shape[1] // 2:
                                            data.graph_type_idx_ranges['ppi'][1] + data.edge_index.shape[1] // 2]

    ####################################################################################################################
    # dpi edges
    ####################################################################################################################

    # Retrieve edge indices, classes and scores
    dpi_edges = data.edge_index[:, data.graph_type_idx_ranges['dpi'][0]:data.graph_type_idx_ranges['dpi'][1]]

    # Same thing with reversed edges
    reversed_dpi_edges = data.edge_index[:, data.graph_type_idx_ranges['dpi'][0] + data.edge_index.shape[1] // 2:
                                            data.graph_type_idx_ranges['dpi'][1] + data.edge_index.shape[1] // 2]

    return torch.cat((ppi_edges, dpi_edges, reversed_ppi_edges, reversed_dpi_edges), dim=1)