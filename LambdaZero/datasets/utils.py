import torch
from torch_geometric.utils import degree


def graph_add_rsqrt_degree_norm(graph):
    """
    Calculate normalization coefficients for message aggregation.
    Flow direction is assumed to be 'target_to_source', note that default in torch_geometric.nn.MessagePassing is 'source_to_target'.
    Normalization over square root of the node degree renders sum of messages to be 0-mean unit-variance distributed,
    under the assumption that individual messages can be viewed as independent samples from 0-mean unit-variance distribution.

    Args:
        graph - torch_geometric.data.Data

    Output:
        graph (modified in-place) - torch_geometric.data.Data
    """
    # assumed flow: ''
    origin_nodes, _ = graph.edge_index  # origin, neighbor
    node_degrees = degree(origin_nodes, num_nodes=graph.x.size(0), dtype=torch.get_default_dtype())
    graph.norm = node_degrees[origin_nodes].rsqrt()  # 1 / sqrt(degree(i))
    return graph


def graph_add_directions_and_distances(graph):
    """
    Calculate relative directions (unit vectors) and absolute distances between connected nodes of the graph based on nodes' positions.

    Args:
        graph - torch_geometric.data.Data

    Output:
        graph (modified in-place) - torch_geometric.data.Data
    """
    # direction
    origin_pos = graph.pos[graph.edge_index[0]]
    neighbor_pos = graph.pos[graph.edge_index[1]]
    rel_vec = (neighbor_pos - origin_pos).type(torch.get_default_dtype())
    graph.rel_vec = torch.nn.functional.normalize(rel_vec, p=2, dim=-1)
    # distance
    graph.abs_distances = rel_vec.norm(dim=1)
    return graph


def graph_select_targets(graph, target_idx):
    # TODO: selecting 0 for QM9, do something better for more general case
    graph.y = graph.y[0, target_idx].type(torch.get_default_dtype())
    return graph
