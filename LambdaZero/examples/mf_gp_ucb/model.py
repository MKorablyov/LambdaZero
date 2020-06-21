import torch

class ValNetwork(torch.nn.Module):
    def __init__(self, architecture=[5,2,1]):
        super().__init__()
        seq = []
        in_size = architecture[0]
        for out_size in architecture[1:]:
            seq.append(torch.nn.Linear(in_features=in_size,out_features=out_size))
            seq.append(torch.nn.ReLU())
            in_size = out_size
        self.seq = torch.nn.Sequential(
                *seq[:-1]
        )
    def forward(self, obs):
        return self.seq(obs)

