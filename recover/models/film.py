import torch
from torch.nn import Sequential
from torch.nn import functional as F
from recover.models.mlp_film_generator import MlpFiLMGenerator

def build_mlp_film(config):
    generator = MlpFiLMGenerator(config)
    return FiLM(generator)

class FiLM(torch.nn.Module):
    def __init__(self, generator):
        super().__init__()
        self.generator = generator

    def forward(self, x, class_conditionals):
        gamma, beta = self.generator(class_conditionals)
        return (gamma * x) + beta

