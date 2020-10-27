from torch_geometric.nn import GlobalAttention, Set2Set
import torch

def global_add_pool(x):
    return x.sum(dim=0)

def global_max_pool(x):
    return x.max(dim=0)

def global_mean_pool(x):
    return x.mean(dim=0)

# Below are wrappers around a couple PyG global pooling techniques which
# just serve to make them all compatible with our interface

class GlobalAttention(GlobalAttention):
    def __init__(self, residual_layers_dim, global_pool_att_lyrs=[]):
        layer_dims = self._get_layer_dims(residual_layers_dim, global_pool_att_lyrs)

        layers = []
        for i in range(len(layer_dims) - 1):
            layer.append(torch.nn.Linear(layer_dims[i], layer_dims[i + 1]))

        mlp = torch.nn.Sequential(*layers)

        super().__init__(nn=mlp)

    @classmethod
    def _get_layer_dims(cls, residual_layers_dim, global_pool_att_lyrs):
        if len(global_pool_att_lyrs) > 0:
            layers = global_pool_att_lyrs
            layers.insert(0, residual_layers_dim)

            if layers[-1] != 1:
                layers.append(1)

        else:
            layers = [residual_layers_dim, 1]

        return layers

    def forward(self, x):
        # Tensor used by PyG's GlobalAttention to scatter node embeddings.
        # In our case, there will (so far) only be one graph in the batch,
        # so we'll just set batch as zeros here.
        batch = torch.zeros((x.shape[0],), device=x.device)
        return super().forward(x, batch)

class Set2Set(Set2Set):
    def __init__(self, residual_layers_dim, num_set2set_steps, set2set_res_lyrs=1):
        self.in_channels = residual_layers_dim
        super().__init__(residual_layers_dim, num_set2set_steps, set2set_res_lyrs)

    @property
    def output_dim(self):
        return self.in_channels * 2

    def forward(self, x):
        # Tensor used by PyG's Set2Set to scatter node embeddings.
        # In our case, there will (so far) only be one graph in the batch,
        # so we'll just set batch as zeros here.
        batch = torch.zeros((x.shape[0],), device=x.device)
        return super().forward(x, batch)

