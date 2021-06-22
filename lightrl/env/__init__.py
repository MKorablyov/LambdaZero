import cv2
from gym.envs.registration import registry, register, make, spec
from lightrl.env import gym_wrappers

register(
    id='BlockMolEnvGraph-v1',
    entry_point='lightrl.env.block_mol_graph_v1:BlockMolEnvGraph_v1',
    kwargs={},
)
