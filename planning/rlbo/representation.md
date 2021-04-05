### Evaluation
Systematic generalization 
Scaffold-based split IE: KNN in the fingerprint space
Independent dataset ChEMBL with experimental data
Flag: Include optional dock pose in training

### Model
Nequip model (exact implementation)  
PNA  
Channel Mixing  
Hierarchical   
Hierarchical (clusters are building blocks) + order messages in higher group 
Hierarchical -> substructure counting
E3NN
Tensor Product
Directional MPNN (aggregation order)
Scattering transform GNN molecule (for short and long-range interactions)
XYZ cutoff / bonds as edges in message passing
Transformer including from M1

### (Re)Train faster
Oversample good molecules, but train with a larger dataset
Warmstart: initialize with good pretrained weights (multiplied by 0.9)

### Actor
Right now we don't have a separate GNN to train one
We could create Boltzmann Search with 3 actions ahead and predict that

### Learn docking pose
learn docking pose as well
There is no feedback from docking 3D to the RL algo; at smallest GNN could predict some atom features related to binding; 
maybe xyz coordinates
learn poses from only energies (link). We could eventually train a full docking algorithm but coarse-grained to the
junction tree.