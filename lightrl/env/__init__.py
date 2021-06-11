from gym.envs.registration import registry, register, make, spec

register(
    id='BlockMolEnvGraph-v1',
    entry_point='lightrl.env.block_mol_graph_v1:BlockMolEnvGraph_v1',
    kwargs={},
)
