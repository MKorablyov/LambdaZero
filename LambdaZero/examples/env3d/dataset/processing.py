from collections import Callable
from copy import copy
from plistlib import Dict

import numpy as np
import ray
from rdkit import Chem
from torch_geometric.data import Data

from LambdaZero.chem import mpnn_feat, torch
from LambdaZero.inputs import _mol_to_graph


@ray.remote
def env3d_proc(smi: str, props: Dict, pre_filter: Callable, pre_transform: Callable):
    """
    This method processes the raw data written in a pandas.feather file and
    turns it into pytorch geometric graph data. This method is meant to be
    used  with class

        LambdaZero.inputs.inputs_op.BrutalDock

    as the argument "proc_func".

    The form of this function follows the template given by

        LambdaZero.inputs.inputs_op._brutal_dock_proc

    This method is essentially boilerplate wrapping around "get_graph_from_properties",
    which takes care of the business logic.

    Args:
        smi (str): the smiles string
        props (list): list of properties
        pre_filter (Callable): filter function to remove bad data
        pre_transform (Callable): transform function to modify the data

    Returns:
        graph (torch_geometric.data.data.Data): a graph data object
    """
    try:
        graph = get_graph_from_properties(smi, props)
    except Exception as e:
        return None
    if pre_filter is not None and not pre_filter(graph):
        return None
    if pre_transform is not None:
        graph = pre_transform(graph)
    return graph


def get_graph_from_properties(smiles: str, properties_dictionary: Dict) -> Data:
    """

    Args:
        smiles (str): the smiles string of the molecule
        properties_dictionary (Dict): graph properties.

    Returns:
        graph (torch_geometric.data.data.Data): a graph data object

    """
    expected_properties = {
        "coord",
        "n_axis",
        "attachment_node_index",
        "attachment_angle",
        "attachment_block_index",
    }

    assert (
        set(properties_dictionary.keys()) == expected_properties
    ), "The properties are not consistent with the env3d dataset. Review code."

    props = dict(properties_dictionary)  # make a local copy to not affect the input

    # The coordinates are stored as np.tobytes in the feather file because pyArrow is finicky.
    # We must convert back to a numpy array, and deal with flat arrays
    flat_coord = np.frombuffer(props["coord"])
    props["coord"] = flat_coord.reshape(len(flat_coord) // 3, 3)
    props["n_axis"] = torch.from_numpy(np.frombuffer(props["n_axis"])).unsqueeze(0)

    mol = Chem.MolFromSmiles(smiles)
    Chem.SanitizeMol(mol)

    # get the mpnn features, but NOT the coordinates; we have them already.
    atmfeat, _, bond, bondfeat = mpnn_feat(mol, ifcoord=False)

    coord = props.pop("coord")
    graph = _mol_to_graph(atmfeat, coord, bond, bondfeat, props)

    return graph


def transform_concatenate_positions_to_node_features(graph: Data) -> Data:
    """
    This function concatenates the node positions to the node features. It is intended to be
    used as a "transform" function in the constructor of InMemoryDataset-like classes.

    Args:
        graph (Data): input graph, assumed to have node features (property x) and node positions (property pos)

    Returns:
        transformed_graph (Data): a transformed version of the graph, where the positions are concatenated to the
                                  node features.
    """

    transformed_graph = copy(graph)
    transformed_graph.x = torch.cat([graph.x, graph.pos], axis=1)

    return transformed_graph
