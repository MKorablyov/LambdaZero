import pytest
import numpy as np
import torch

from LambdaZero.datasets.temp_brunos_work.graph_score_normalizer import GraphScoreNormalizer


@pytest.fixture
def mean():
    np.random.seed(12312)
    return np.random.rand()


@pytest.fixture
def std():
    np.random.seed(22)
    return np.random.rand()


def test_graph_score_normalizer(random_molecule_data, mean, std):

    graph_score_normalizer = GraphScoreNormalizer(mean, std)

    initial_score = random_molecule_data.gridscore

    expected_normalized_score = (initial_score-mean)/std

    normalized_data = graph_score_normalizer.normalize_score(random_molecule_data)

    assert torch.all(torch.eq(normalized_data.gridscore, expected_normalized_score))

    computed_molecule_data = graph_score_normalizer.denormalize_score(normalized_data)

    assert torch.all(torch.eq(computed_molecule_data.gridscore, random_molecule_data.gridscore))




