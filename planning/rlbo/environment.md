### Bugs
Environment exits on max size molecules; we should adjust the action space instead as in env_v4
Action "delete Block" is written based on splitting molecule into halves deleting leaf block might be easier for model
When environment restarts, it would reset MDP in a very strange way leaving many molecules of 1-3 blocks
This is kind of a bug when oracle dies due to error; handle errors in proxy


### Molecule Graph serialization API
Serialize molecule with some standard API; current is handcrafted and hard to use


### Compositional action
The action space is too large and mostly meaningless IE: 550 * 30 = 15000 actions in some cases. We should try to 
factorize this into choose_stem, choose_block


### Action tree API
At smallest, we could make action to allow agent return to previous state(s).
We also want to allow the agent to carry on the computation between states to allow to develop local strategies. 

### Action Embeddings
We might be able to achieve better results with action embeddings (because of the large action space)


### We may be able to achieve high 3D structure diversity
Add SP3-rich natural compounds