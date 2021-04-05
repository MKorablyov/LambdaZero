### Evaluation
We need to develop a measure of power of the method "IE equivalent in STDs". The starting point could be already best
molecules from a large Boltzmann Search. 


### Boltzmann Search
This includes random/greedy too


### Maximum Likelihood
Maximum likelihood model on half-dataset


### Restart buffer
Accept based on most recent acquisition value
Could help due to make learned policy easier
Sampling starting state according to temperature
Use Maximum Likelihood Model to generate starting states


### Dense Reward
Reward (such as EI) on each step
We want RL agent to be able to observe oracle uncertainties on each step
We would be able to tweak RL horizon as we may want a long one
We could completely remove episode boundaries (IE: assign credit for old actions)

### Cheap training
To train faster we could start RL with imitation learning on good molecules from some
unbiased model such as maximum likelihood or Boltzmann Search
Generate artificial trajectories between two good molecules for imitation with TD algorithms


### Stabilize changes in the Acquisition function
When ACQ changes, sample some important past trajectories but with new ACQ reward
Observe some characteristics of local space; maybe observe future uncertainties under a few fixed policies
Another idea is to sample policy 5 times from a given state 1 step ahead and summarize; We could use GNN/message passing
on action space to summarize
Another idea is to return closest 5 molecules to current state from memory


### Reward Scaling
for linear improvement RL reward should be exponential but that's already included in env
exp
sigmoid
reward buffer (relative reward rankings)


### Other
Try if local entropy on actions in RL is helpful 
Check if larger/smaller number of SGD batch size helps (I am afraid there is much variance)
Just evaluate every single method and derivatives which are already available in the Ray RlLib - PPO is old
We could warm-start RL policy with some imitation learning on Boltzmann search and make iterations/testing much faster
Run RL agent with just 3-4 steps
policy version of AlphaZero - make work
Population of agents (PPO or DQN) to help policy exploration