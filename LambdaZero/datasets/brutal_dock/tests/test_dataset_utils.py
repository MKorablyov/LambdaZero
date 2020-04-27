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


def test_get_scores_statistics(dockscores, training_graphs):

    expected_mean = np.mean(dockscores)
    expected_std = np.std(dockscores)

    dataloader = DataLoader(training_graphs, batch_size=10)

    computed_mean, computed_std = get_scores_statistics(dataloader)

    np.testing.assert_almost_equal(computed_mean, expected_mean)
    np.testing.assert_almost_equal(computed_std, expected_std)
