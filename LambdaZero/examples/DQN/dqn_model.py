'''
 I honestly just dug myself in a hole and buried myself in it and kept digging


 Why is this entering a different policy's get_q_values... returns (1, 2107)


 How did torch_graph_models (PPO) handle the stop_logits and done_logits... this doesn't seem to be here :/ in the 2107

 Somehow calculated a get_state_score too... without ours...


 Issue 1: It's using this model's forward, but is not using this model's get_q_value_dist and not using get_state_value
 Issue 2: num_outputs for this one is preset to 256... I've got no idea where this parameter is set or what it means -- I'm guessing the embedding size?
                -- Ah maybe I don't need to calculate this get_q_values and it'll do the "actions for me"...
                   - First off, I need to replace this to override the "impossible" actions (action_mask)
                -- But strangely, it seems to want output of forward() to be of size (None, num_outputs) = (?, 256) -- why though? 256 is a standard batch size but it's not a standard matrix notiation, typically (256, 1) 

 Issue 3: PPO seems to work? The num_outputs seems to be 2107 (# of actions as expected... intended)
'''



from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

import os

from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
import numpy as np
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import try_import_torch
from torch_geometric.nn import GINEConv
from torch_geometric.nn import NNConv
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, Batch
import torch.nn.functional as F

from LambdaZero.utils import RunningMeanStd

# import torch and torch.nn using ray utils
torch, nn = try_import_torch()

# Running pdb - set # of workers to 0
class GraphMolDQN_thv1(DQNTorchModel, nn.Module): # Not sure what _thv1 meant
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name) # How is num_outputs determined? -- Why is this supposed to be 256?
        nn.Module.__init__(self)
        # self.obs_space = obs_space # This is Box(18307,)
        # self.action_space = action_space # This is Discrete(2107) -- something like this...
        # self.num_outputs = num_outputs # What is this number of outputs supposed to mean?


        self.preprocessor = get_preprocessor(obs_space.original_space)(obs_space.original_space)
        self.max_steps = obs_space.original_space["num_steps"].n
        self.max_blocks = action_space.max_blocks # default: 7
        self.max_branches = action_space.max_branches # default: 20
        self.num_blocks = action_space.num_blocks # default: 105

        self.space = obs_space.original_space['mol_graph']
        self.dim_hidden = 128 # hack to get this working -- for some reason the output needs to be 256 -- to check
        self.state_embedder = MPNNet_Parametric(self.space.num_node_feat,
                                       kw.get('num_hidden', self.dim_hidden),
                                       self.num_blocks)
        
        # self.dim_action = self.max_branches * self.num_blocks + 6 + 1 # Calculated based on debugging... PPO - product of branches and blocks, break_logit, and stop_logit
        self.dim_action = self.max_blocks + self.max_branches * self.num_blocks # This is the calculation used in the block mol environment - future just copy from action_space

        
        self.function_base = nn.Sequential(
                                    nn.Linear(256, 256) #temp hardcoded
                            )
        self.q_function = nn.Sequential(
                                self.function_base,
                                nn.Linear(256, self.dim_action) 
                            )
        self.value_network = nn.Sequential(
                                self.function_base,
                                nn.Linear(256, 1)
                            )

        self.action_mask_dict = {}


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

        state_embeddings, data = self.state_embedder(data)
        self.action_mask_dict[state_embeddings] = action_mask # Hack to get this working

        # Leo: Need to think about "stopping"/"testing" -- this should be an "action" in itself
        # Set of Actions: (Picking a stem and adding a part, Stopping and evaluating, Breaking logit -- prolly means delete)

        return state_embeddings, state 

    def get_state_value(self, model_out): # Leo: This shouldn't be necessary for normal DQN...
        # but this references it https://docs.ray.io/en/master/rllib-concepts.html?highlight=dqn#building-policies-in-tensorflow
        """Returns the state value prediction for the given state embedding."""
        print("This is used only for dueling-Q")
        value_output =  self.value_network(model_out)
        return value_output

    def get_q_value_distributions(self, model_out): # TODO: Temporarily ignoring this "get_state_value"
        """Returns distributional values for Q(s, a) given a state embedding.

        Override this in your custom model to customize the Q output head.

        Args:
            model_out (Tensor): (state_embeddings) Embedding from the model layers.

        Returns:
            (action_scores, logits, dist) if num_atoms == 1, otherwise
            (action_scores, z, support_logits_per_action, logits, dist)
        """

        state_embeddings = model_out
        action_scores = self.q_function(state_embeddings) # Takes as input the embedding? Where does this embedding come from?

        action_mask = self.action_mask_dict[state_embeddings]
        self.action_mask_dict[state_embeddings] = None # Temporary hack that needs to be removed later...

        action_scores = action_scores + action_mask*10000 # TODO -- Perhaps instead of setting Q(s, a) = 0 for illegal actions, we can simply ignore them... -- and maybe the function'll be smoother.

        # I'm not sure what the point of these logits are... (besides being used in the dueling DQN)
        # And why were these logits all 1 in the original code?
        logits = torch.unsqueeze(torch.ones_like(action_scores), -1)
        assert action_mask[action_mask > 0].shape == action_scores[action_scores > 1000].shape
        return action_scores, logits, logits 

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))



class MPNNet_Parametric(nn.Module): # this should output a Q(s, a) instead.

    def __init__(self, num_feat=14, dim=64, num_out_per_stem=105):
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


