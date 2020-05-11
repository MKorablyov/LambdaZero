import functools
from os import path

import numpy as np
from scipy import special

from LambdaZero.environments import molecule
from LambdaZero.examples import config as cfg


def step_env(rng, env, state, action_mask, temperature, mol_eval):
    actions = np.where(action_mask)[0]
    values = []
    next_states = []
    observations = []
    for i, a in enumerate(actions):
        env.set_state(state)
        obs, _, _, info = env.step(a)
        values.append(mol_eval(info["molecule"]))
        next_states.append(env.get_state())
        observations.append(obs)

    probs = special.softmax(np.divide(values, temperature))
    a = rng.choice(actions.shape[0], p=probs)
    return values[a], next_states[a], obs[a]


def boltzmann_search(max_steps, mol_eval, temperature=1., stop_condition=None,
                     top_k=1):
    config = cfg.mol_blocks_v4_config()
    config["env_config"]["reward_config"]["device"] = "cpu"
    env = config["env"](config["env_config"])

    obs = env.reset()
    state = env.get_state()

    values = []
    states = []
    rng = np.random.default_rng()

    for i in range(max_steps):
        action_mask = obs["action_mask"]
        val, state, obs = step_env(rng, env, state, action_mask, temperature,
                                   mol_eval)
        values.append(val)
        states.append(state)

        if stop_condition is not None:
            stop_condition.update(val, state, obs)
            if stop_condition.should_stop():
                break

    top_idx = np.argsort(values)[-top_k:]
    return values[top_idx], states[top_idx]


LAMBDAZERO_ROOT = "/home/gehring/PycharmProjects/LambdaZero/"
DATASET_DIR = path.join(LAMBDAZERO_ROOT, "Datasets")

rew_eval = molecule.PredDockReward(
    load_model=path.join(DATASET_DIR, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
    natm_cutoff=[45, 50],
    qed_cutoff=[0.2, 0.7],
    soft_stop=False,
    exp=None,
    delta=False,
    simulation_cost=0.0,
    device="cpu",
)
rew_eval.reset()
rew_eval = functools.partial(rew_eval,
                             env_stop=False,
                             simulate=True,
                             num_steps=1)

if __name__ == "__main__":
    print(boltzmann_search(100, rew_eval, 1.0, None, 3))
