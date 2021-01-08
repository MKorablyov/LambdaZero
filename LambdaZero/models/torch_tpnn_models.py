import torch
import torch.nn as nn

from torch_geometric.nn import Set2Set, global_mean_pool
#from e3nn.point.message_passing import TensorPassingContext, TensorPassingLayer

from functools import partial


class TPNN_v0(torch.nn.Module):
    def __init__(self, max_z, representations, equivariant_model, radial_model, gate, pooling, fc):
        super().__init__()

        assert pooling in ['avg', 'set2set'], "Pooling can be either 'avg' or 'set2set'"

        radial_model = partial(radial_model['type'], **radial_model['config'])
        gate = partial(gate['type'], **gate['config'])
        self.equivariant_model = equivariant_model(representations, radial_model, gate)

        self.max_z = max_z
        self.emb_size = self.equivariant_model.input_representations[0][0][0]  # 1st layer -> scalar representation -> multiplicity
        self.out_size = self.equivariant_model.output_representations[-1][0][0]

        self.emb_layer = torch.nn.Embedding(num_embeddings=self.max_z, embedding_dim=self.emb_size, padding_idx=0)

        self.fc = None
        if pooling == 'set2set':
            self.pooling = Set2Set(self.out_size, processing_steps=3)
            if fc:
                self.fc = nn.Sequential(
                    nn.Linear(2 * self.out_size, self.out_size),
                    nn.ReLU(),
                    nn.Linear(self.out_size, 1)
                )
        else:  # avg
            self.pooling = global_mean_pool
            if fc:
                self.fc = nn.Sequential(
                    nn.Linear(self.out_size, self.out_size),
                    nn.ReLU(),
                    nn.Linear(self.out_size, 1)
                )

    def forward(self, graph):
        features = self.emb_layer(graph.z)
        features = self.equivariant_model(graph.edge_index, features, graph.abs_distances, graph.rel_vec, graph.norm)
        output = self.pooling(features, graph.batch)
        if self.fc is not None:
            output = self.fc(output)
        return output


# class TPNN_ResNet(TensorPassingContext):
#     def __init__(self, representations, radial_model, gate):
#         super().__init__(representations)
#         self.model = torch.nn.ModuleList([
#             TensorPassingLayer(Rs_in, Rs_out, self.named_buffers_pointer, radial_model, gate)
#             for (Rs_in, Rs_out) in zip(self.input_representations[:-1], self.output_representations[:-1])
#         ])
#         # no gate on last layer
#         self.model.append(TensorPassingLayer(self.input_representations[-1], self.output_representations[-1], self.named_buffers_pointer, radial_model))
#
#     def forward(self, edge_index, features, abs_distances, rel_vec, norm):
#         features = self.model[0](edge_index, features, abs_distances, rel_vec, norm)
#         for layer in self.model[1:-1]:
#             features = (features + layer(edge_index, features, abs_distances, rel_vec, norm)).mul(0.7071)  # o[n] = (o[n-1] + f[n]) / sqrt(2)
#         features = self.model[-1](edge_index, features, abs_distances, rel_vec, norm)
#         return features


class TPNN_Unet(TensorPassingContext):
    def __init__(self, representations, radial_model, gate):
        super().__init__(representations)
        self.model = torch.nn.ModuleList([
            TensorPassingLayer(Rs_in, Rs_out, self.named_buffers_pointer, radial_model, gate) for (Rs_in, Rs_out) in zip(self.input_representations[:-1], self.output_representations[:-1])
        ])
        # no gate on last layer
        self.model.append(TensorPassingLayer(self.input_representations[-1], self.output_representations[-1], self.named_buffers_pointer, radial_model))

        self.n_equivariant_layers = len(self.model)
        self.n_arm_layers = (self.n_equivariant_layers - 1) // 2
        self.has_mid_layer = ((self.n_equivariant_layers - 1) % 2) == 1

    def forward(self, edge_index, features, abs_distances, rel_vec, norm):
        model_iter = iter(self.model)
        features_list = []

        # left U arm
        for _ in range(self.n_arm_layers):
            layer = next(model_iter)
            features = layer(edge_index, features, abs_distances, rel_vec, norm)
            features_list.append(features)

        # mid layer
        if self.has_mid_layer:
            layer = next(model_iter)
            features = layer(edge_index, features, abs_distances, rel_vec, norm)

        # right U arm
        for _ in range(self.n_arm_layers):
            layer = next(model_iter)
            features = (layer(edge_index, features, abs_distances, rel_vec, norm) + features_list.pop()).mul(0.7071)

        layer = next(model_iter)
        features = layer(edge_index, features, abs_distances, rel_vec, norm)
        return features
