import numpy as np
import torch
from torch import nn
import torch_scatter

# ==================================================================
# Zhaocheng: Pseudo code. Never verified.

class GiantGraphDataset(torch.utils.data.Dataset):
    # this class should provide the adjacency tensor of the graph
    # usually stored in COO sparse matrix style
    # 
    # edge_list: (num_edge, 2) for homogenous graph or (num_edge, 3) for relational graph
    # edge_weight: (num_edge)
    # 
    # a COO sparse matrix can be easily constructed from edge_list and edge_weight
    
    def __init__(self, data, num_node, num_relation):
        self.edge_list = data
        
        # Manually flatten 3D COO index to 2D COO index
        row = data[0] * num_relation + data[2]
        col = data[1]
        index = torch.stack([row, col])
        value = torch.ones(len(data))
        self.adjacency = torch.sparse.FloatTensor(index, value, size=(num_node * num_relation, num_node))

# General Message Passing style implementation of GNNs
# Similar to the spirit of PyG / DGL
class MessagePassingBase(nn.Module):

    def message(self, graph, input):
        raise NotImplementedError

    def aggregate(self, graph, message):
        raise NotImplementedError

    def message_and_aggregate(self, graph, input):
        message = self.message(graph, input)
        update = self.aggregate(graph, message)
        return update

    def combine(self, input, update):
        raise NotImplementedError

    def forward(self, graph, input):
        update = self.message_and_aggregate(graph, input)
        output = self.combine(input, update)
        return output
 
# A RGCN implementation based on third party torch_scatter. Similar to the implementation in PyG. 
#
# We may also implement using the native torch.spmm. torch.spmm is even faster since it wraps CuSparse and doesn't explicit create the messages.
# However, torch.spmm only works for 2D tensor.
# We need to reshape the adjacency (num_node, num_relation, num_node) into (num_node * num_relation, num_node) in propagation.
# It could be less readable.
class RelationalGraphConv(MessagePassingBase):

    eps = 1e-10

    def __init__(self, input_dim, output_dim, num_relation, batch_norm=False, activation="relu"):
        super(RelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation
        self.self_loop = nn.Linear(input_dim, output_dim)
        self.linear = nn.Linear(num_relation * input_dim, output_dim)

    def message(self, graph, input):
        node_in = graph.edge_list[:, 1]
        message = input[node_in]
        return message

    def aggregate(self, graph, message):
        assert graph.num_relation == self.num_relation
        
        # edge_list[:, 0] is node_out
        # edge_list[:, 2] is relation
        node_out = graph.edge_list[:, 0] * self.num_relation + graph.edge_list[:, 2]
        edge_weight = graph.edge_weight.unsqueeze(-1)
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) / \
                 (scatter_add(edge_weight, node_out, dim=0, dim_size=graph.num_node * self.num_relation) + self.eps)
        return update.view(graph.num_node, -1)

    def combine(self, input, update):
        x = self.linear(update) + self.self_loop(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        output = self.activation(x)
        return output

    
class DrugCombination(nn.Module):
    
    def __init__(self, model, dataset)
        # For giant graph, there is no need to use a dataloader.
        # We store the graph in a buffer and keep it **always** on GPU.
        self.gnn_model = model
        self.drug_model = # some model that generates drug representations from protein representations
        self.register_buffer("train_edges", dataset.train_edges)
        self.register_buffer("adjacency", dataset.adjacency)
    
    def sample(self):
        # return positive & negative samples from self.train_edges
        # this can be done on GPU
        return node_in, node_out
    
    def forward(self): 
        # An even smarter way is to accumulate the gradients for large graphs.
        # And update gnn_model & drug_model less frequently than combination_model.
        protein = self.gnn_model(self.adjacency)
        drug = self.drug_model(protein)
        
        x, y = self.sample()
        pred = self.combination_model(drug[x], drug[y])
        
        return pred
    
# Zhaocheng: End of Pseudo code.
# ==================================================================

class RGCN_data:
    def __init__(self, some):
        print("init!!!!!!!!!!!1")
        self.some = some
        pass


class RGCN_v1(torch.nn.Module, RGCN_data):

    def __init__(self):
        #self.data = data

        super(RGCN_data, self).__init__()
        super(RGCN_v1, self).__init__()

        #super(torch.nn.Module, self).__init__()
    #super(Envelope, self).__init__()


    pass


DEFAULT_CONFIG = {
    "num_relations":5,
    "num_nodes":5000,
    "num_edges": 50000

}


def _fake_data(num_relations, num_nodes, num_edges ):
    """

    :param num_relations:
    :param num_nodes:
    :param num_edges:
    :return: data = [edge_i, edge_j, relation]
    """

    edge_i = torch.randint(num_nodes, size=[num_edges, 1])
    edge_j = torch.randint(num_nodes, size=[num_edges,1])
    relation = torch.randint(num_relations, size=[num_edges,1])

    data = torch.cat([edge_i, edge_j, relation],dim=1)
    return data

def _train_epoch():
    pass



if __name__ == "__main__":
    config = DEFAULT_CONFIG
    data = _fake_data(config["num_relations"], config["num_nodes"], config["num_edges"])
    print(data)
        
    GiantGraphDataset(data, config["num_relations"])
            
    RGCN_v1().to("cuda")

