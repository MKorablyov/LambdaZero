import pytest
import numpy as np
from torch_geometric.data import Data

from torch_geometric.data import DataLoader

from LambdaZero.datasets.brutal_dock.dataset_utils import get_scores_statistics


@pytest.fixture
def dockscores():
    np.random.seed(12312)
    return np.random.rand(1000)


@pytest.fixture
def training_graphs(dockscores):

    list_graphs = []
    for dockscore in dockscores:
        graph = Data()
        graph.dockscore = dockscore
        list_graphs.append(graph)

    return list_graphs


@pytest.mark.parametrize("number_of_molecules", [10])
def test_get_scores_statistics(list_dockscores, list_random_molecules):

    dockscore_array = list_dockscores.numpy().flatten()

    expected_mean = np.mean(dockscore_array)
    expected_std = np.std(dockscore_array)

    dataloader = DataLoader(list_random_molecules, batch_size=10)

    computed_mean, computed_std = get_scores_statistics(dataloader)

    np.testing.assert_almost_equal(computed_mean, expected_mean)
    np.testing.assert_almost_equal(computed_std, expected_std)
