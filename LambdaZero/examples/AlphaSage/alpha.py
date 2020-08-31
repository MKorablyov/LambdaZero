# todo: if done: make reward for traj; reward in classification
# todo: walk classifier:
#       get shared model in Ray in env (maybe quite hard)
#       make a backward hook to train classifier on each epoch


# MoleculeWalkEnv(num_blocks=5, walk_len=10):
#   obs = {action_embeddings, walk_embeddings, mask}
#   self.mol_mdp = MolMDP()
#   self.mol = mol_mdp.walk()
#   self.atom_embeddings = []
#   self.coursor = 0

# def step(action)
#
#   return obs, reward, done, info