*****
Exploring Exploration
*****

Exploration is a key challenge in reinforcement learning. In particular for LambdaZero, when searching for possible molecules we want to maximize exploration in the space of molecules. This will allow us to produce a set of candidate molecules with (possibly) diverse structure and properties.

Approaches: 

A. Random Network Distillation
Random Network Distillation is an exploration technique that uses the prediction error of a predictor network with respect to a fixed random network to reward the agent for visting novel states. This is added as a bonus reward to the agent. More specifically, this prediction error encourages the agent to visit previously unseen states as the prediction loss is higher for novel states. 
In the experiments so far, RND does not provide any improvements over the PPO baselines. 

B. Local Exploration 
This is another intrinsic reward based method. Here at each time step the agent receives a bonus based on the distance from the molecule it started with. We compute the distance between the molecular fingerprints. 
This approach provides higher max rewards compared to the PPO baselines.  


Other approaches tried: 
1. Maintain a persistent buffer of previously encountered molecules, and add the average distance of the current molecule from all previous modeluces as a bonus. => Poor results so far.
2. In the above setting instead of adding a bonus reward, only allow molecules to the buffer if the distance is greater than a certain threshold. => Poor results so far.