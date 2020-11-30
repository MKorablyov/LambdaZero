from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

# import torch and torch.nn using ray utils
torch, nn = try_import_torch()

# Running pdb - set # of workers to 0
class GraphMolDQN(TorchModelV2, nn.Module): # Not sure what _thv1 meant
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)
        self.max_steps = obs_space.original_space["num_steps"].n
        self.max_blocks = action_space.max_blocks
        self.max_branches = action_space.max_branches
        self.num_blocks = action_space.num_blocks

        self.space = obs_space.original_space['mol_graph']
        self.dim_hidden = 64
        self.state_embedder = MPNNet_Parametric(self.space.num_node_feat,
                                       kw.get('num_hidden', self.dim_hidden),
                                       self.num_blocks,
                                       self.rnd)
        
        self.dim_action = self.num_blocks + 1 + 1 # Calculated based on action space in PPO code -- needs confirmation
        self.q_function = nn.Linear(self.dim_hidden * 2 + self.dim_action, 1) # Linear layer temporarily -- todo
        self._value_out = None

        self.action_mask = None


    def forward(self, input_dict, state, seq_lens): # Torch geometric - takes list of edges
        # Leo: This is possibly supposed to output an embedding...
        #           This returns a state embedding (refer to "compute_q_values" in dqn_torch_policy in ray rllib)! 

        obs = input_dict['obs']
        device = obs["mol_graph"].device

        # Due to the way RLLib forces encodeing observations into
        # fixed length vectors, this part is quite expensive (~timings
        # on big batch), first there's a useless
        # uint8->float->cpu->gpu->cpu->uint8 (~3ms), then we unpack
        # each graph individually which involves a decompress call and
        # more memory copying (~12ms), then we create a Batch from the
        # "data list" of graphs which involves stacking and
        # incrementing arrays (~4ms) and send it back to gpu (<.5ms).
        enc_graphs = obs["mol_graph"].data.cpu().numpy().astype(np.uint8)
        graphs = [self.space.unpack(i) for i in enc_graphs]
        num_steps = obs["num_steps"]
        action_mask = obs["action_mask"] # Hack to allow get_q_value_dist access to this

        data = fast_from_data_list(graphs) # Leo: Check -- Can I run it w/ debugging -- pdb. Num_workers = 1?
        data = data.to(device)
        # </end of expensive unpacking> The rest of this is the
        # forward pass (~5ms) and then a backward pass + update (which
        # I can't measure directly but seems to take at most ~20ms)

        import pdb; pdb.set_trace() # Check that the output of the state_embedder is 2 * dim_hidden
        state_embeddings, data = self.state_embedder(data)
        # Leo: Need to think about "stopping"/"testing" -- this should be an "action" in itself
        # Set of Actions: (Picking a stem and adding a part, Stopping and evaluating, Breaking logit -- prolly means delete)


        # stop_logit = scalar_outs[:, 1] # Leo: what's this stop_logit and isn't this just [:, 1]. Maybe just calling docking?
        # # Ends rollout -- and give agent rew.
        # add_logits = data.stem_preds.reshape((data.num_graphs, -1))
        # break_logits = data.jbond_preds.reshape((data.num_graphs, -1))

        # actor_logits = torch.cat([stop_logit,
        #                           add_logits,
        #                           break_logits], 1)

        return (state_embeddings, action_mask), state # I'm guessing I should be returning the actual value: Q(s, a)

    # def value_function(self):
    #     return self._value_out # Leo: TODO -- needs replacing

    # def get_state_value(self, model_out): # Leo: This shouldn't be necessary for normal DQN...
    # # but this references it https://docs.ray.io/en/master/rllib-concepts.html?highlight=dqn#building-policies-in-tensorflow
    #     """Returns the state value prediction for the given state embedding."""

    #     return self.value_module(model_out)

    def get_q_value_distributions(self, model_out):
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Args:
            model_out (Tensor, Tensor): (state_embeddings, action_mask) Embedding from the model layers.

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """
        # Embedding... now what about the possible actions? -- Are these one hot encodings? -- TODO -- pretty sure yes -- Run on Beluga w/ num_workers = 0, import pdb; pdb.set_trace()
        
        import pdb; pdb.set_trace()
        state_embeddings, action_mask = model_out
        # Should I be appending the onehot encoding of actions? While ignoring the impossible actions.
        model_in = state_embeddings # Leo -- should be a concat of state action pairs (I think) TODO ####### 
        action_scores = self.q_function(model_in) # Takes as input the embedding? Where does this embedding come from?

        masked_actions = (1. - action_mask)

        action_scores = action_scores * masked_actions # TODO -- Perhaps instead of setting Q(s, a) = 0 for illegal actions, we can simply ignore them -- and maybe the function'll be smoother.




        # I'm not sure what the point of these logits are... (besides being used in the dueling DQN)
        # And why were these logits all 1 in the original code?
        logits = torch.unsqueeze(torch.ones_like(action_scores), -1)

        return action_scores, logits, logits 

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))



class MPNNet_Parametric(nn.Module): # this should output a Q(s, a) instead.

    def __init__(self, num_feat=14, dim=64, num_out_per_stem=105, rnd=False):
        super().__init__()

        self.lin0 = nn.Linear(num_feat, dim)
        self.num_ops = num_out_per_stem

        net = nn.Sequential(nn.Linear(4, dim), nn.LeakyReLU(), nn.Linear(dim, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        # self.node2stem = nn.Sequential(
        #     nn.Linear(dim, dim), nn.LeakyReLU(), nn.Linear(dim, num_out_per_stem)) # the thing is... these are actions!
        # self.node2jbond = nn.Sequential(
        #     nn.Linear(dim, dim), nn.LeakyReLU(), nn.Linear(dim, 1)) # the thing is... these are actions!

        

        self.set2set = Set2Set(dim, processing_steps=1) # Leo: What's this? -- Look into this...


        # Now needs a FFN for (embedding, action)
        # self.dim_action = dim_action
        # self.num_input = num_feat + (num_out_per_stem + 1) + 1 # (s, a) 14 + 105 + 1 + 1 -- one hot action encoding


        # 2 = [simulate logit] - value, logit for whether to end or not
        # Leo: Not sure why it's dim * 2 though
        # self.lin_out = nn.Linear(dim * 2 + dim_action, 1) # Leo: What's a simulate logit? --- this is an action too!

    def forward(self, data):
        out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        # 2 main functions for GNN - pooling and aggregation
        for i in range(6): # -- \# of message passes. What's the GRU for -- propagating messages 
            # edge_attr - features for the edge
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr)) # Conv -- generates the messages - takes the adjacency matrix and performs conv.
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous()) # a -> b - m      b -> c - m2
            out = out.squeeze(0)

        # data.stem_preds = self.node2stem(out[data.stem_atmidx])
        # data.jbond_preds = self.node2jbond(out[data.jbond_atmidx.flatten()]) \
        #                        .reshape((data.jbond_atmidx.shape)) \
        #                        .mean(1) # mean pooling of the 2 jbond atom preds

        out = self.set2set(out, data.batch) # Leo: Not sure what this set2set does... and what data.batch is
        # out = self.lin_out(out)
        return out, data # this should return an embedding of size dim x 2?


# This is mostly copied from torch_geometric, but with a few checks
# and generalities removed, as well as some speed improvements
def fast_from_data_list(data_list,
                        inckeys=set(['stem_atmidx','edge_index','jbond_atmidx'])):
    r"""Constructs a batch object from a python list holding
    :class:`torch_geometric.data.Data` objects.
    The assignment vector :obj:`batch` is created on the fly.
    """

    keys = data_list[0].keys

    batch = Batch()
    batch.__data_class__ = data_list[0].__class__
    batch.__slices__ = {key: [0] for key in keys}

    for key in keys:
        batch[key] = []

    cumsum = [0] * len(keys)
    batch.batch = []
    for j, key in enumerate(keys):
        cat_dim = 1 if key == 'edge_index' else 0
        slc = batch.__slices__[key]
        bk = batch[key]
        for i, data in enumerate(data_list):
            item = data[key]
            if cumsum[j] > 0:
                item = item + cumsum[j]
            slc.append(item.shape[cat_dim] + slc[-1])
            if key in inckeys:
                cumsum[j] += data.x.shape[0]
            bk.append(item)
            if j == 0:
                batch.batch.append(torch.full((data.x.shape[0], ), i, dtype=torch.long))

    for key in batch.keys:
        item = batch[key][0]
        cd = data_list[0].__cat_dim__(key, item)
        batch[key] = torch.cat(batch[key],
                               dim=cd)

    return batch.contiguous()






# torch, nn = try_import_torch()


# class DQNTorchModel(TorchModelV2, nn.Module):
#     """Extension of standard TorchModelV2 to provide dueling-Q functionality.
#     """

#     def __init__(
#             self,
#             obs_space,
#             action_space,
#             num_outputs,
#             model_config,
#             name,
#             *,
#             q_hiddens=(256, ),
#             dueling=False,
#             dueling_activation="relu",
#             num_atoms=1,
#             use_noisy=False,
#             v_min=-10.0,
#             v_max=10.0,
#             sigma0=0.5,
#             # TODO(sven): Move `add_layer_norm` into ModelCatalog as
#             #  generic option, then error if we use ParameterNoise as
#             #  Exploration type and do not have any LayerNorm layers in
#             #  the net.
#             add_layer_norm=False):
#         """Initialize variables of this model.

#         Extra model kwargs:
#             q_hiddens (List[int]): List of layer-sizes after(!) the
#                 Advantages(A)/Value(V)-split. Hence, each of the A- and V-
#                 branches will have this structure of Dense layers. To define
#                 the NN before this A/V-split, use - as always -
#                 config["model"]["fcnet_hiddens"].
#             dueling (bool): Whether to build the advantage(A)/value(V) heads
#                 for DDQN. If True, Q-values are calculated as:
#                 Q = (A - mean[A]) + V. If False, raw NN output is interpreted
#                 as Q-values.
#             dueling_activation (str): The activation to use for all dueling
#                 layers (A- and V-branch). One of "relu", "tanh", "linear".
#             num_atoms (int): If >1, enables distributional DQN.
#             use_noisy (bool): Use noisy layers.
#             v_min (float): Min value support for distributional DQN.
#             v_max (float): Max value support for distributional DQN.
#             sigma0 (float): Initial value of noisy layers.
#             add_layer_norm (bool): Enable layer norm (for param noise).
#         """
#         nn.Module.__init__(self)
#         super(DQNTorchModel, self).__init__(obs_space, action_space,
#                                             num_outputs, model_config, name)

#         self.dueling = dueling
#         self.num_atoms = num_atoms
#         self.v_min = v_min
#         self.v_max = v_max
#         self.sigma0 = sigma0
#         ins = num_outputs # What is this number of outputs? -- Should be a scalar for normal DQN but a distribution for distributional DQN

#         self.value_module = MPNNet_Parametric(self.space.num_node_feat, # Q(s, a)
#                                        kw.get('num_hidden', 64),
#                                        self.num_blocks,
#                                        self.rnd)


#     def get_q_value_distributions(self, model_out):
#         """Returns distributional values for Q(s, a) given a state embedding.

#         Override this in your custom model to customize the Q output head.

#         Args:
#             model_out (Tensor): Embedding from the model layers.

#         Returns:
#             (action_scores, logits, dist) if num_atoms == 1, otherwise
#             (action_scores, z, support_logits_per_action, logits, dist)
#         """
#         action_scores = self.advantage_module(model_out). # Takes as input the embedding? Where does this embedding come from?

#         if self.num_atoms > 1:
#             # Distributional Q-learning uses a discrete support z
#             # to represent the action value distribution
#             z = torch.range(0.0, self.num_atoms - 1, dtype=torch.float32)
#             z = self.v_min + \
#                 z * (self.v_max - self.v_min) / float(self.num_atoms - 1)

#             support_logits_per_action = torch.reshape(
#                 action_scores, shape=(-1, self.action_space.n, self.num_atoms))
#             support_prob_per_action = nn.functional.softmax(
#                 support_logits_per_action)
#             action_scores = torch.sum(z * support_prob_per_action, dim=-1)
#             logits = support_logits_per_action
#             probs = support_prob_per_action
#             return action_scores, z, support_logits_per_action, logits, probs
#         else:
#             # This should be non-distributional...
#             logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
#             return action_scores, logits, logits # Why are all the logits 1? How is this useful at all... I suppose more useful for other extensions of DQN?

#     def get_state_value(self, model_out): # Leo: Is this necessary?
#         """Returns the state value prediction for the given state embedding."""

#         return self.value_module(model_out)


