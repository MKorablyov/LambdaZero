import numpy as np
import torch
import torch_scatter


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

    RGCN_v1().to("cuda")

