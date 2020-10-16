import torch
import torch.nn as nn

from torch_geometric.nn import Set2Set, global_mean_pool

from e3nn.point.message_passing import TensorPassingHomogenous, TensorPassingContext, TensorPassingLayer
from e3nn.radial import GaussianRadialModel, CosineBasisModel
from e3nn.point.gate import Gate
from e3nn.non_linearities.rescaled_act import swish, tanh, sigmoid, relu
from functools import partial


class TPNN_v1(TensorPassingHomogenous):
    def __init__(self, representations, use_set2set=True):
        hidden_size = representations[-1][0][0]
        self.use_set2set = use_set2set
        radial_model = partial(GaussianRadialModel, min_radius=0.7, max_radius=3.2, number_of_basis=10, h=100, L=3, act=swish)
        gate = partial(Gate, scalar_act=tanh, tensor_act=tanh)
        super().__init__(representations, radial_model, gate)
        if use_set2set:
            self.pooling = Set2Set(hidden_size, processing_steps=3)
            self.fully_connected = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )
        else:
            self.pooling = global_mean_pool
            self.fully_connected = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, graph):
        hidden_features = super().forward(graph)
        pooled_features = self.pooling(hidden_features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output


class TPNN_v2(TensorPassingHomogenous):
    def __init__(self, representations, **kwargs):
        emb_size = 16
        hidden_size = representations[-1][0][0]
        input_size = representations[0][0][0]
        representations[0][0] = (emb_size, 0, 0)
        min_radius = kwargs.get('min_radius', 1.1)
        max_radius = kwargs.get('max_radius', 2.3)
        number_of_basis = kwargs.get('number_of_basis', int(10*(max_radius - min_radius)))
        radial_model = partial(GaussianRadialModel, min_radius=min_radius, max_radius=max_radius, number_of_basis=number_of_basis, h=100, L=3, act=swish)
        gate = partial(Gate, scalar_act=tanh, tensor_act=tanh)
        super().__init__(representations, radial_model, gate)

        self.emb = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.Tanh()
        )
        self.pooling = Set2Set(hidden_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
        )

    def forward(self, graph):
        graph.x = self.emb(graph.x)
        hidden_features = super().forward(graph)
        pooled_features = self.pooling(hidden_features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output


class TPNN_ResNet(TensorPassingContext):
    def __init__(self, max_z, representations, radial_model, gate):
        super().__init__(representations)
        self.max_z = max_z
        self.emb_size = self.input_representations[0][0][0]  # 1st layer -> scalar representation -> multiplicity
        self.out_size = self.output_representations[-1][0][0]

        self.emb_layer = torch.nn.Embedding(num_embeddings=self.max_z, embedding_dim=self.emb_size)

        self.model = torch.nn.ModuleList([
            TensorPassingLayer(Rs_in, Rs_out, self.named_buffers_pointer, radial_model, gate) for (Rs_in, Rs_out) in zip(self.input_representations[:-1], self.output_representations[:-1])
        ])

        # no gate on last layer
        self.model.append(TensorPassingLayer(self.input_representations[-1], self.output_representations[-1], self.named_buffers_pointer, radial_model))

        self.pooling = Set2Set(self.out_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
            nn.Linear(2 * self.out_size, self.out_size),
            nn.ReLU(),
            nn.Linear(self.out_size, 1)
        )

    def forward(self, graph):
        edge_index = graph.edge_index
        abs_distances = graph.abs_distances
        rel_vec = graph.rel_vec
        norm = graph.norm

        embedding = self.emb_layer(graph.z)

        features = self.model[0](edge_index, embedding, abs_distances, rel_vec, norm)
        for layer in self.model[1:-1]:
            features = (features + layer(edge_index, features, abs_distances, rel_vec, norm)).mul(0.7071)  # o[n] = (o[n-1] + f[n]) / sqrt(2)
        features = self.model[-1](edge_index, features, abs_distances, rel_vec, norm)

        pooled_features = self.pooling(features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output


class TPNN_ResNet_Avg(TensorPassingContext):
    def __init__(self, max_z, avg_n_atoms, representations, radial_model, gate):
        super().__init__(representations)
        self.max_z = max_z
        self.register_buffer('norm', torch.tensor([avg_n_atoms], dtype=torch.float64).rsqrt())
        self.emb_size = self.input_representations[0][0][0]  # 1st layer -> scalar representation -> multiplicity
        self.out_size = self.output_representations[-1][0][0]

        self.emb_layer = torch.nn.Embedding(num_embeddings=self.max_z, embedding_dim=self.emb_size, padding_idx=0)

        self.model = torch.nn.ModuleList([
            TensorPassingLayer(Rs_in, Rs_out, self.named_buffers_pointer, radial_model, gate) for (Rs_in, Rs_out) in zip(self.input_representations[:-1], self.output_representations[:-1])
        ])

        # no gate on last layer
        self.model.append(TensorPassingLayer(self.input_representations[-1], self.output_representations[-1], self.named_buffers_pointer, radial_model))

        self.pooling = global_mean_pool

    def forward(self, graph):
        edge_index = graph.edge_index
        abs_distances = graph.abs_distances
        rel_vec = graph.rel_vec
        norm = self.norm  # norm is the same for all nodes

        embedding = self.emb_layer(graph.z)

        features = self.model[0](edge_index, embedding, abs_distances, rel_vec, norm)
        for layer in self.model[1:-1]:
            features = features + layer(edge_index, features, abs_distances, rel_vec, norm)  # o[n] = o[n-1] + f[n]
        features = self.model[-1](edge_index, features, abs_distances, rel_vec, norm)

        output = self.pooling(features, graph.batch)
        return output


class TPNN_Unet(TensorPassingContext):
    def __init__(self, max_z, representations, radial_model, gate):
        super().__init__(representations)
        self.max_z = max_z
        self.emb_size = self.input_representations[0][0][0]  # 1st layer -> scalar representation -> multiplicity
        self.out_size = self.output_representations[-1][0][0]

        self.emb_layer = torch.nn.Embedding(num_embeddings=self.max_z, embedding_dim=self.emb_size)

        self.model = torch.nn.ModuleList([
            TensorPassingLayer(Rs_in, Rs_out, self.named_buffers_pointer, radial_model, gate) for (Rs_in, Rs_out) in zip(self.input_representations[:-1], self.output_representations[:-1])
        ])

        # no gate on last layer
        self.model.append(TensorPassingLayer(self.input_representations[-1], self.output_representations[-1], self.named_buffers_pointer, radial_model))

        self.pooling = Set2Set(self.out_size, processing_steps=3)
        self.fully_connected = nn.Sequential(
            nn.Linear(2 * self.out_size, self.out_size),
            nn.ReLU(),
            nn.Linear(self.out_size, 1)
        )

        self.n_equivariant_layers = len(self.model)
        self.n_arm_layers = (self.n_equivariant_layers - 1) // 2
        self.n_base_layers = (self.n_equivariant_layers - 1) % 2

    def forward(self, graph):
        edge_index = graph.edge_index
        abs_distances = graph.abs_distances
        rel_vec = graph.rel_vec
        norm = graph.norm

        model_iter = iter(self.model)
        features_list = []

        features = self.emb_layer(graph.z)

        # left U arm
        for _ in range(self.n_arm_layers):
            layer = next(model_iter)
            features = layer(edge_index, features, abs_distances, rel_vec, norm)
            features_list.append(features)

        # base - can be either one layer or none
        for _ in range(self.n_base_layers):
            layer = next(model_iter)
            features = layer(edge_index, features, abs_distances, rel_vec, norm)

        # right U arm
        for _ in range(self.n_arm_layers):
            layer = next(model_iter)
            features = layer(edge_index, features, abs_distances, rel_vec, norm) + features_list.pop()

        layer = next(model_iter)
        features = layer(edge_index, features, abs_distances, rel_vec, norm)

        pooled_features = self.pooling(features, graph.batch)
        output = self.fully_connected(pooled_features)
        return output
