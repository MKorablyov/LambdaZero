import torch
from torch_geometric.utils import degree
from torch_geometric.utils import tree_decomposition
from torch_geometric.data import Data
from rdkit import Chem


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
    origin_nodes, _ = graph.edge_index  # origin, neighbor
    num_nodes = graph.x.size(0) if graph.x is not None else graph.z.size(0)
    node_degrees = degree(origin_nodes, num_nodes=num_nodes, dtype=torch.get_default_dtype())
    graph.norm = node_degrees[origin_nodes].rsqrt()  # 1 / sqrt(degree(i))
    return graph


def graph_add_distances(graph):
    """
    Calculate relative directions (unit vectors) and absolute distances between connected nodes of the graph based on nodes' positions.

    Args:
        graph - torch_geometric.data.Data

    Output:
        graph (modified in-place) - torch_geometric.data.Data
    """
    origin_pos = graph.pos[graph.edge_index[0]]
    neighbor_pos = graph.pos[graph.edge_index[1]]
    rel_vec = (neighbor_pos - origin_pos).type(torch.get_default_dtype())
    graph.abs_distances = rel_vec.norm(dim=1)
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


def mol_graph_add_bond_type(graph):
    # assumes that order of nodes is the same as the order of atoms in smiles string
    # assumes that edge index was also constructed from .GetBonds(), and bi-directionality is achieved via staking of flipped edge_index
    mol = Chem.MolFromSmiles(graph.smiles)
    bond_types = torch.tensor([bond.GetBondType().real for bond in mol.GetBonds()], dtype=torch.int64)
    graph.bond_type = torch.cat([bond_types, bond_types])
    return graph


# modified from https://github.com/rusty1s/himp-gnn/blob/0a506bb6ecc66b5ffc29b9b229c28a805adb06c1/transform.py
def mol_graph_add_junction_tree(graph):
    mol = Chem.MolFromSmiles(graph.smiles)
    clique_edge_index, node2clique_index, num_cliques, x_clique = tree_decomposition(mol, return_vocab=True)
    graph = DataWithJunctionTree(**{k: v for k, v in graph})  # for proper handling of collation
    graph.clique_edge_index = clique_edge_index
    graph.node2clique_index = node2clique_index
    graph.num_cliques = num_cliques
    graph.x_clique = x_clique
    return graph


class DataWithJunctionTree(Data):
    """
    The point of this class is too ensure correct index shifts upon collation.
    """
    def __inc__(self, key, item):
        if key == 'clique_edge_index':
            return self.x_clique.size(0)
        elif key == 'node2clique_index':
            return torch.tensor([[self.num_nodes], [self.x_clique.size(0)]])
        else:
            return super().__inc__(key, item)
