# molecular environment

# molMDP().random_walk() -> mol_graph, r-groups
# choose r-group randomly
# molMDP().make_subs(r-group) -> mol_graph * 105
# MPNN(mol_graph, r_group) -> logit * 105           # actor
# MPNN(mol_graphs) -> label * 105                   # critic