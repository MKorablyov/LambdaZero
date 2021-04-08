### Initialize start buffer / MPNN with best from Boltzmann Search
Boltzmann Search should result in much stronger binders compared to Zinc. We could use both best molecules from 
Boltzmann Seach as well as MPNN/acquisition_function to initialize RL. We should also quantufy the rate of improvement 
For off-policy methods we could pretrain policy on this Boltzmann Search as well


### Validate how diverse are generated molecules
Quantify pairwise Tanimoto diversity (of RL and acquired batch)


### Likelihoods
Split good molecules from Boltzmann into train/test
How much error comes from policy, how much from model with uncertainty