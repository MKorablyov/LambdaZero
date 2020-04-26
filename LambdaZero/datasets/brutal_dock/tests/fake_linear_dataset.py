import torch
from torch.utils.data import Dataset


class FakeLinearDataset(Dataset):
    def __init__(
        self,
        seed: int,
        number_of_points: int,
        input_dimension: int,
        output_dimension: int,
    ):
        torch.manual_seed(seed)
        factor = 1.0
        self.number_of_points = number_of_points
        self.weights = factor * torch.rand(
            input_dimension, output_dimension, requires_grad=False
        )
        self.bias = factor * torch.rand(output_dimension, requires_grad=False)
        self.x = torch.rand(number_of_points, input_dimension)
        self.y = torch.mm(self.x, self.weights) + self.bias
        self.y = self.y.view(-1, output_dimension)

    def __len__(self):
        return self.number_of_points

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.x[idx], self.y[idx]

