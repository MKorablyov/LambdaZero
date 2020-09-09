import torch

class FiLM(torch.nn.Module):
    def __init__(self, num_cell_lines, num_features):
        '''
        Arguments
        ---------
        num_cell_lines : int
            The number of cell lines that we have to condition for
        num_features : int
            The number of features that we will need to condition
        '''
        super().__init__()
        self.gammas = Parameter(1/100 * torch.randn(num_cell_lines, num_features))
        self.betas = Parameter(1/100 * torch.randn(num_cell_lines, num_features))

    def forward(self, x, cell_lines):
        return self.gammas[cell_lines] * x + self.betas[cell_lines]

