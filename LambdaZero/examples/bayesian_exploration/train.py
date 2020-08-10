import argparse
import time
import ray
from ray import tune
from env import Atari, Preprocessor
from ray.rllib.models.catalog import ModelCatalog
from LambdaZero.examples.AlphaZero.core.alpha_zero_trainer import AlphaZeroTrainer
# from ray.rllib.agents.trainer_template import build_trainer
# from ray.rllib.policy.sample_batch import SampleBatch
# from ray.rllib.policy.torch_policy_template import build_torch_policy
from model import AZNetwork
# from LambdaZero.models.ppo_rnd import PPORNDTrainer
parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--num-cpus", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    
    ray.init()
    ModelCatalog.register_custom_preprocessor("custom", Preprocessor)
    # ModelCatalog.register_custom_model("conv_model", ConvNetModel)
    tune.run(
        AlphaZeroTrainer,# "contrib/AlphaZero", # AlphaZeroTrainer, # "contrib/AlphaZero",# PPORNDTrainer,
        # stop={"training_iteration": args.stop_iters},
        config={
            "env": Atari,
            "env_config": {
                "env_name": "BreakoutNoFrameskip-v4",
            },
            "model": {
                "grayscale": True,
                # "zero_mean": True,
                "custom_model": AZNetwork,
                # "custom_model_config": {
                #     "in_channels": 3,
                #     "feature_dim": 512,
                # },
                "custom_preprocessor": "custom",
            },
            # "num_workers": 7,
            # "num_envs_per_worker": 8,
            # "num_gpus_per_worker": 0.075,
            # "num_gpus": 1,
            "framework": "torch",
            # "model": {
            #     # "custom_model": BayesianVisionNetwork,
            #     # "custom_model_config": {
            #     #     "rnd_weight": 1,
            #     #     "rnd_output_dim": 512
            #     # }
            # },
            "rollout_fragment_length": 8,
            "train_batch_size": 8,
            "sgd_minibatch_size": 8,
            "lr": 1e-4,
            "num_sgd_iter": 1,
            "mcts_config": {
                "puct_coefficient": 1.5,
                "num_simulations": 2,
                "temperature": 1.0,
                "dirichlet_epsilon": 0.20,
                "dirichlet_noise": 0.03,
                "argmax_tree_policy": False,
                "add_dirichlet_noise": True,
            },
            "ranked_rewards": {
                "enable": True,
                "percentile": 75,
                "buffer_max_length": 1000,
                # add rewards obtained from random policy to
                # "warm start" the buffer
                "initialize_buffer": True,
                "num_init_rewards": 2,
            },

            # "preprocessor_pref": "deepmind"
            # "lambda": 0.95,
            # "gamma": 0.99,
            # "kl_coeff": 0.5,
            #  "lr": 1e-4,
            # "clip_rewards": True,
            # "clip_param": 0.1,
            # # "vf_clip_param": 10.0,
            # "entropy_coeff": 0.001,
            # "train_batch_size": 5000,
            # # "sample_batch_size": 100,
            # "rollout_fragment_length": 128,
            # # "sgd_minibatch_size": 500,
            # # "batch_mode": "truncate_episodes",
            # "observation_filter": "NoFilter",
            # "num_sgd_iter": 4,
        })