import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch_scatter

# ==================================================================
# Zhaocheng: Pseudo code.

class GiantGraphDataset(torch.utils.data.Dataset):
    # this class should provide the adjacency tensor of the graph
    # usually stored in COO sparse matrix style
    # 
    # edge_list: (num_edge, 2) for homogenous graph or (num_edge, 3) for relational graph
    # edge_weight: (num_edge)
    # 
    # a COO sparse matrix can be easily constructed from edge_list and edge_weight
    
    def __init__(self, edge_list, num_node, num_relation):
        self.edge_list = edge_list
        
        # Manually flatten 3D COO index to 2D COO index
        row = edge_list[:, 0] * num_relation + edge_list[:, 2]
        col = edge_list[:, 1]
        index = torch.stack([row, col])
        value = torch.ones(len(edge_list))
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
# We may also implement using the native torch.spmm.
# torch.spmm is even faster since it wraps CuSparse and doesn't explicit create the messages.
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
    
    def __init__(self, model, dataset):
        super(DrugCombination, self).__init__()
        # For giant graph, there is no need to use a dataloader.
        # We store the graph in a buffer and keep it **always** on GPU.
        self.gnn_model = model
        # some model that generates drug representations from protein representations
        self.drug_model = None
        self.register_buffer("edge_list", dataset.edge_list)
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
    



DEFAULT_CONFIG = {
    "num_relations": 5,
    "num_nodes": 5000,
    "num_edges": 50000,
    "feature_dim": 100,
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
    edge_list = _fake_data(config["num_relations"], config["num_nodes"], config["num_edges"])
    #node_feature = torch.randn(config["num_nodes"], config["feature_dim"])
    print("edge list: %s" % edge_list)
    #print("node feature: %s" % node_feature)
        
    dataset = GiantGraphDataset(edge_list, config["num_nodes"], config["num_relations"])
    
    model = nn.Sequential(
        RelationalGraphConv(config["feature_dim"], config["feature_dim"], config["num_relations"]),
        RelationalGraphConv(config["feature_dim"], config["feature_dim"], config["num_relations"]),
    )
    
    task = DrugCombination(model, dataset).cuda()
    print(task)
    #print(task.forward())
    # RGCN_v1().to("cuda")