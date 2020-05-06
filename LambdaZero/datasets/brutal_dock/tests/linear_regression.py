from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, number_of_inputs: int, number_of_outputs: int):
        super().__init__()
        self.linear = nn.Linear(number_of_inputs, number_of_outputs)

    def forward(self, x):
        return self.linear(x)
