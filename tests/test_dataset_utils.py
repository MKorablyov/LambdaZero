import pytest
import numpy as np
from torch_geometric.data import Data

from torch_geometric.data import DataLoader

from LambdaZero.datasets.temp_brunos_work.dataset_utils import get_scores_statistics


@pytest.fixture
def gridscores():
    np.random.seed(12312)
    return np.random.rand(1000)


@pytest.fixture
def training_graphs(gridscores):

    list_graphs = []
    for gridscore in gridscores:
        graph = Data()
        graph.gridscore = gridscore
        list_graphs.append(graph)

    return list_graphs


@pytest.mark.parametrize("number_of_molecules", [10])
def test_get_scores_statistics(list_gridscores, random_molecule_dataset):

    gridscore_array = list_gridscores.numpy().flatten()

    expected_mean = np.mean(gridscore_array)
    expected_std = np.std(gridscore_array)

    dataloader = DataLoader(random_molecule_dataset, batch_size=10)

    computed_mean, computed_std = get_scores_statistics(dataloader)

    np.testing.assert_almost_equal(computed_mean, expected_mean)
    np.testing.assert_almost_equal(computed_std, expected_std)
