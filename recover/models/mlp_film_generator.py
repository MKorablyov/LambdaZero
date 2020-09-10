import torch
from torch.nn import Sequential
from torch.nn import functional as F

class MlpFiLMGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        channels = config['film_hidden_channels']
        channels.insert(0, config['film_in_channels'])
        channels.append(2 * config['film_out_channels'])

        modules = []
        for i in range(1, len(channels)):
            modules.append(torch.nn.Linear(channels[i - 1], channels[i]))
            if i == len(channels) - 1:
                modules.append(torch.nn.ReLU())

        self.w = Sequential(modules)
        self.out_channels = config['film_out_channels']

    def forward(self, x):
        out = self.w(x)
        gamma, beta = out[:self.out_channels], out[self.out_channels:]

        return gamma, beta

