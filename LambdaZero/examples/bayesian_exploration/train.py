import argparse
import time
import ray
from ray import tune
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_template import build_torch_policy
from model import BayesianVisionNetwork
from LambdaZero.models.ppo_rnd import PPORNDTrainer
parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--num-cpus", type=int, default=0)


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()
    time.sleep(240) 
    tune.run(
        PPORNDTrainer,
        # stop={"training_iteration": args.stop_iters},
        config={
            "env": "MontezumaRevenge-v0",
            "num_workers": 7,
            "num_gpus_per_worker": 0.075,
            "num_gpus": 1,
            "framework": "torch",
            "model": {
                "custom_model": BayesianVisionNetwork,
                "custom_model_options": {
                    "rnd_weight": 1,
                    "rnd_output_dim": 256
                }
            },
            "lambda": 0.95,
            "kl_coeff": 0.5,
             "lr": 1e-4,
            "clip_rewards": True,
            "clip_param": 0.1,
            "vf_clip_param": 10.0,
            "entropy_coeff": 0.0001,
            "train_batch_size": 5000,
            "rollout_fragment_length": 128,
            "sgd_minibatch_size": 500,
            "num_sgd_iter": 4,
        })