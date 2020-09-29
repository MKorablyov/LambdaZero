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
    def __init__(self, channels, t, eps, diff_aggr_h, diff_aggr_l, walk_aggr_h, walk_aggr_l):
        """
        :param channels: channels in channels out
        :param t: number of diffusion steps
        :param eps: noise to add to the probability of diffusion
        """
        super(DiffGCN, self).__init__()
        self.channels = channels
        self.t = t
        self.eps = eps
        self.diff_aggr = GRUAggr(channels[0], channels[1], diff_aggr_h, diff_aggr_l)
        self.walk_aggr = GRUAggr(channels[0], channels[1], walk_aggr_h, walk_aggr_l)

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
        walk_embeds = v[:, None, :]
        walks_logp = []
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
            walk_embeds_t = torch.cat([walk_embeds_t, v_t[:, None, :]],dim=1)
            walk_logp_t = self.diff_aggr(walk_embeds_t)[:, 0]
            # rescale walk probs
            init_vs = repeat_cat(walks[:,0], adj_slic_t[:,1])
            with torch.no_grad(): # todo - make sure this norm is ok
                norm = torch_scatter.scatter_logsumexp(walk_logp_t, init_vs, dim=0, dim_size=num_nodes)
            norm = repeat_cat(norm,adj_slic_t[:,1])
            walk_logp_t = walk_logp_t - norm
            walk_p_t = torch.exp(walk_logp_t)
            #print(v.shape, v_t.shape, walk_logp_t.shape, walk_p_t.shape)
            # print(scatter(walk_p_t,init_vs, reduce="sum", dim=0, dim_size=num_nodes))
            # add_noise
            # todo: categorical noise and why
            noise = self.eps * torch.randn(walk_p_t.shape,device=v.device)
            noise_walk_p_t = walk_p_t + noise
            _, walk_t = torch_scatter.scatter_max(noise_walk_p_t, init_vs, dim=0, dim_size=num_nodes)
            walks_logp.append(walk_logp_t[walk_t])
            walk_t = adj_t[1][walk_t]
            # update walks and walk embeddings
            walks = torch.cat([walks, walk_t[:,None]], dim=1)
            walk_embeds = torch.cat([walk_embeds,v[walk_t,:][:,None,:]],dim=1)
        walks_logp = torch.stack(walks_logp,dim=1)
        return walks, walk_embeds, walks_logp

    def aggregate(self, walks, walk_embeds): # [e,d] -> [v,d]
        return self.walk_aggr(walk_embeds)

    def forward(self, node_attr, edge_index, slices):
        walks, walk_embeds, walks_p = self.diffuse(node_attr, edge_index, slices)
        # print(walk_embeds.sum(dim=[2]))
        if np.random.randn() > 0.995:
            self._test_walks(node_attr, edge_index, walks, walk_embeds)
        v_out = self.aggregate(walks, walk_embeds)

        # todo: I need to maximize prob of good walks, and minimize prob of bad walks
        # todo: I want loss from trajectory to be same as misclassification
        #
        # [s0, s1, s2, s3] # reward == loss
        # r = - loss
        # p[s0, s1] -> - loss
        # p[s0, s1, s2] -> -loss

        return v_out, walks_p