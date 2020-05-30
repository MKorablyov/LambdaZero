# disc
# 1. Adjacency matrix                               [num_side_effects x num_drugs x num_drugs ]
# 2. node embeddings Z                              [num_drugs x Rd]
# 3. importance matrices D                          [num_side_effects x eye([Rd x Rd])]
# 4. global importance matrix R                     [Rd x Rd]
# 5. side_effect_stitch_id -> adjacency_matrix_idx  []
# 6. train/test split                               [num_side_effects x num_drugs x num_drugs ]
# 7. side effect was used for training              [num_side_effects]                              # postpone


# def compute_AUC(P_edges, A_r):
    # return AUC


# def predict(side_effect_stitch):
#   side_effect_stitch -> side_effect_idx
#   D_r = importance_matrix[side_effect_idx]
#   iterate (num_nodes):
#       iterate (num_nodes)
        # p_edge =  sigmoid( Zi @ D_r @ R @ D_r @ Zj)
    # p_edges
#   return AUC