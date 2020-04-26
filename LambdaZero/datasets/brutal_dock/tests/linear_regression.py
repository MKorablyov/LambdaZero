from torch import nn


class LinearRegression(nn.Module):
    def __init__(self, number_of_inputs: int, number_of_outputs: int):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(number_of_inputs, number_of_outputs)

    def normalized_forward(self, x):
        return self.linear(x)
