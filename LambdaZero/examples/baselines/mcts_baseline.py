import sys
from rdkit import Chem

import ray
from ray import tune
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils import merge_dicts

from LambdaZero.utils.mcts import MCTS, Node, RootParentNode
from LambdaZero.examples.baselines import config
from LambdaZero.examples.baselines.boltzmann_baseline import EnvEval
from LambdaZero.models.torch_graph_models import GraphMolActorCritic_thv1
import LambdaZero.utils

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

#@ray.remote
class MCTS_opt():
    def __init__(self, config):
        config["env_config"]["reward_config"]["device"] = "cpu"
        self.env = config["env"](config["env_config"])

        self.model = config["model"]
        self.mcts = MCTS(self.model, config["mcts_config"])

        self.steps = config["steps"]
        self.docking = config["docking"]
        self.env_eval = EnvEval(self.env, config["env_eval_config"])

    def optimize_molecule(self):  # __call__(self):
        self.obs = self.env.reset()
        for i in range(self.steps):
            self.step()
        # return None

    def step(self):
        env_state = self.env.get_state()
        tree_node = Node(
            state=env_state,
            obs=self.obs,
            reward=0,
            done=False,
            action=None,
            parent=RootParentNode(env=self.env),
            mcts=self.mcts
        )
        mcts_policy, action, tree_node = self.mcts.compute_action(tree_node)
        self.obs, reward, _, info = self.env.step(action)
        if self.docking:
            reward = self.env_eval(docking=self.docking)
        log_vals = {**{"reward":reward, "molecule":Chem.MolToSmiles(info["molecule"].mol)}, **info["log_vals"]}
        # self.env.set_state(self.env_state)
        tune.report(log_vals)
        # return log_vals

def mcts_opt_wrapper(config):
    mcts_opt = MCTS_opt(config)
    mcts_opt.optimize_molecule()

DEFAULT_CONFIG = {
    "mcts_opt_config": {
        "mcts_config": {
            "puct_coefficient": 1.5,
            "num_simulations": 800,
            "temperature": 1.0,
            "dirichlet_epsilon": 0.020,
            "dirichlet_noise": 0.003,
            "argmax_tree_policy": False,
            "add_dirichlet_noise": True,
            "policy_optimization": False,
        },
        "model": {
            "custom_model": "GraphMolActorCritic_thv1",
        },
    },
    "summaries_dir": summaries_dir,
    "memory": 60 * 10 ** 9,
    "reuse_actors": True,
    "num_samples": 200,
    "resources_per_trial": {
        "cpu": 4,
        "gpu": 0.4,
        # usually, request 40 cpus, 4 gpus (entire node on Beluga)
    },
}

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    else:
        config_name = "mcts_config_000"
    config = getattr(config, config_name)
    config = merge_dicts(DEFAULT_CONFIG, config)

    ray.init(memory=config["memory"])
    ModelCatalog.register_custom_model("GraphMolActorCritic_thv1", GraphMolActorCritic_thv1)

    analysis=tune.run(
        mcts_opt_wrapper, #MCTS_opt
        config=config["mcts_opt_config"],
        reuse_actors=config["reuse_actors"],
        num_samples=config["num_samples"],
        resources_per_trial=config["resources_per_trial"],
        local_dir=config["summmaries_dir"],
    )