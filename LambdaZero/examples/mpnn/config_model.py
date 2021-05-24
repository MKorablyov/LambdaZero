from LambdaZero.models import MPNNWithPlainNNConv, MPNNWithHierarchicalNNConv, MPNNWithPlainPNAConv, MPNNWithHierarchicalPNAConv

_config_shared = {
    "max_z": 54,                # from ZINC20 Database
    "max_edge_value": 22,       # from rdkit.Chem.BondTypes
    "flow": 'target_to_source'  # match tpnn, for tpnn it is important due to msg degree normalization
}

_config_shared_pna = {
    **_config_shared,
    "avg_deg_lin": 2.44,        # from ZINC20 100k subset (molecular graph)
    "avg_deg_log": 1.19,        # from ZINC20 100k subset (molecular graph)
}

_config_node_v0 = {
    "node_emb_size": 8,
    "node_hidden_channels": 64,
    "node_output_channels": 32
}

_config_node_pna_v0 = {
    **_config_node_v0,
    "msg_channels": 64
}

_config_clique_v0 = {
    "max_vocab_value": 4,
    "clique_emb_size": 4,
    "clique_hidden_channels": 32,
    "clique_output_channels": 16,
    "clique_edge_emb_size": 4,
}

_config_edge_v0 = {
    "edge_emb_type": 'blank',
    "edge_emb_size": 8
}

_config_edge_v1 = {
    "edge_emb_type": 'bond_type',
    "edge_emb_size": 8
}

_config_edge_v2 = {
    "edge_emb_type": 'distance',
    "edge_emb_size": 8
}

_config_edge_v3 = {
    "edge_emb_type": 'bond_type+distance',
    "edge_emb_size": 8
}

nnconv_plain_v0_v0 = {
    "model": MPNNWithPlainNNConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_v0,
        **_config_edge_v0,
        **_config_shared
    }
}

nnconv_plain_v0_v1 = {
    "model": MPNNWithPlainNNConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_v0,
        **_config_edge_v1,
        **_config_shared
    }
}

nnconv_plain_v0_v2 = {
    "model": MPNNWithPlainNNConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_v0,
        **_config_edge_v2,
        **_config_shared
    }
}

nnconv_plain_v0_v3 = {
    "model": MPNNWithPlainNNConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_v0,
        **_config_edge_v3,
        **_config_shared
    }
}

nnconv_hmp_v0_v1_v0 = {
    "model": MPNNWithHierarchicalNNConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_v0,
        **_config_edge_v1,
        **_config_clique_v0,
        **_config_shared
    }
}

pnaconv_plain_v0_v0 = {
    "model": MPNNWithPlainPNAConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_pna_v0,
        **_config_edge_v0,
        **_config_shared_pna
    }
}

pnaconv_plain_v0_v1 = {
    "model": MPNNWithPlainPNAConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_pna_v0,
        **_config_edge_v1,
        **_config_shared_pna
    }
}

pnaconv_plain_v0_v2 = {
    "model": MPNNWithPlainPNAConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_pna_v0,
        **_config_edge_v2,
        **_config_shared_pna
    }
}

pnaconv_plain_v0_v3 = {
    "model": MPNNWithPlainPNAConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_pna_v0,
        **_config_edge_v3,
        **_config_shared_pna
    }
}

pnaconv_hmp_v0_v1_v0 = {
    "model": MPNNWithHierarchicalPNAConv,
    "model_config": {
        "n_inter_layers": 5,
        **_config_node_pna_v0,
        **_config_edge_v1,
        **_config_clique_v0,
        **_config_shared_pna
    }
}
