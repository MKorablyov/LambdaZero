import time, os.path as osp
import torch
import numpy as np
from torch_sparse import SparseTensor
import torch_scatter

from aggr import MLP, GRUAggr
from ray import tune

def slice_cat(tensor,slices, dim):
    chunks = [tensor.narrow(dim, slice[0], slice[1]) for slice in slices]
    return torch.cat(chunks,dim=dim)

# todo: torch jit optimize
def repeat_cat(tensor, repeats):
    _once = np.ones(len(tensor.shape),dtype=np.int)[1:]
    chunks = [tensor.narrow(0,i,1).repeat(repeat, *_once) for i,repeat in enumerate(repeats)]
    return torch.cat(chunks,dim=0)


class DiffGCN(torch.nn.Module):
    def __init__(self, channels, t, eps):
        """
        :param channels: channels in channels out
        :param t: number of diffusion steps
        :param eps: noise to add to the probability of diffusion
        """
        super(DiffGCN, self).__init__()
        self.channels = channels
        self.t = t
        self.eps = eps
        self.diff_mlp = MLP([channels[0]*(1+t), 64, 1])
        self.walk_aggr = GRUAggr(channels[0], channels[1], 128, 1)

    def _walk_logp(self, walk_embeds_t):
        "copute probability of each walk on the graph"
        walk_embeds_t = walk_embeds_t.view([walk_embeds_t.shape[0], self.channels[0] * (1+self.t)])
        # unscaled walk probs
        with torch.no_grad():  # diffusion gradient provided separately
            walk_logp_t = self.diff_mlp(walk_embeds_t)[:, 0]
        return walk_logp_t

    def _test_walks(self, node_attr, adj, walks, walk_embeds):
        node_attr, adj = node_attr.cpu().numpy(), adj.cpu().numpy()
        walks, walk_embeds = walks.cpu().numpy(), walk_embeds.cpu().numpy()
        for i,walk in enumerate(walks):
            for j in range(self.t-1):
                in_adj = (adj - np.array([[walk[j]], [walk[j+1]]]))==0
                in_adj = sum(np.logical_and(in_adj[0], in_adj[1]))
                assert in_adj == 1, "walk id not in the adjacency matrix"
                assert np.array_equal(walk_embeds[i][j], node_attr[walk[j]]), "incorrect embedding"
        return None

    def diffuse(self, v, adj, adj_slic):
        num_nodes = v.shape[0]
        # initialize
        walks = torch.arange(num_nodes,device=v.device)[:,None] # [w, 1] --> [w,t]
        walk_embeds = torch.zeros([num_nodes, 1+self.t, self.channels[0]], dtype=torch.float, device=v.device)#[w, t, d]
        walk_embeds[:,0,:] = v

        # diffuse
        for t in range(self.t):
            # find slices of the adjacency matrix
            adj_slic_t = adj_slic[walks[:, -1]]
            # make new adjacency matrix;
            adj_t = slice_cat(adj, adj_slic_t, dim=1)
            # get node_in embeddings
            v_t = v[adj_t[1]]
            # form timestep walk embeds
            walk_embeds_t = repeat_cat(walk_embeds, adj_slic_t[:,1])
            walk_embeds_t[:, 1+t, :] = v_t
            walk_logp_t = self._walk_logp(walk_embeds_t)
            # rescale walk probs
            init_vs = repeat_cat(walks[:,0], adj_slic_t[:,1])
            norm = torch_scatter.scatter_logsumexp(walk_logp_t, init_vs, dim=0, dim_size=num_nodes)
            norm = repeat_cat(norm,adj_slic_t[:,1])
            walk_p_t = torch.exp(walk_logp_t - norm)
            # print(scatter(walk_p_t,init_vs, reduce="sum", dim=0, dim_size=num_nodes))
            # add_noise
            # todo: categorical noise and why
            noise = self.eps * torch.randn(walk_p_t.shape[0],device=v.device)
            walk_p_t = walk_p_t + noise
            _, walks_t = torch_scatter.scatter_max(walk_p_t, init_vs, dim=0, dim_size=num_nodes)
            walks_t = adj_t[1][walks_t]
            # update walks and walk embeddings
            walks = torch.cat([walks, walks_t[:,None]], dim=1)
            walk_embeds[:,1+t, :] = v[walks_t,:]
        return walks, walk_embeds

    def aggregate(self, walks, walk_embeds): # [e,d] -> [v,d]
        return self.walk_aggr(walk_embeds)

    def forward(self, node_attr, edge_index, slices):
        walks, walk_embeds = self.diffuse(node_attr, edge_index, slices)
        # print(walk_embeds.sum(dim=[2]))
        if np.random.randn() > 0.995:
            self._test_walks(node_attr, edge_index, walks, walk_embeds)
        v_out = self.aggregate(walks, walk_embeds)
        return v_out

    # def backward_hook()
        # rewards = F([s0,s1,s2,s3], l2_loss)
        # R * log (prob(a | s))