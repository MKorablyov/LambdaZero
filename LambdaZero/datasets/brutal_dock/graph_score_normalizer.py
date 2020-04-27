from copy import deepcopy

from torch_geometric.data import Data


class GraphScoreNormalizer:
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def normalize_score(self, graph: Data):

        normalized_graph = deepcopy(graph)
        normalized_graph.dockscore = (graph.dockscore-self.mean)/self.std
        return normalized_graph

    def denormalize_score(self, normalized_graph: Data):
        graph = deepcopy(normalized_graph)
        graph.dockscore = self.std*graph.dockscore + self.mean
        return graph