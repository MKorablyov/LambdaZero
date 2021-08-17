import gym
from starlette.requests import Request
import requests

import ray
import ray.rllib.agents.ppo as ppo
from ray import serve

def train_ppo_model():
    trainer = ppo.PPOTrainer(
        config={
            "framework": "torch",
            "num_workers": 0
        },
        env="CartPole-v0",
    )
    # Train for one iteration
    trainer.train()
    trainer.save("/tmp/rllib_checkpoint")
    return "/tmp/rllib_checkpoint/checkpoint_000001/checkpoint-1"

checkpoint_path = train_ppo_model()

@serve.deployment(route_prefix="/cartpole-ppo")
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        self.trainer = ppo.PPOTrainer(
            config={
                "framework": "torch",
                # only 1 "local" worker with an env (not really used here).
                "num_workers": 0,
            },
            env="CartPole-v0")
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]

        action = self.trainer.compute_action(obs)
        return {"action": int(action)}

@serve.deployment(route_prefix="/cartpole-ppo")
class ServePPOModel:
    def __init__(self, checkpoint_path) -> None:
        self.trainer = ppo.PPOTrainer(
            config={
                "framework": "torch",
                # only 1 "local" worker with an env (not really used here).
                "num_workers": 0,
            },
            env="CartPole-v0")
        self.trainer.restore(checkpoint_path)

    async def __call__(self, request: Request):
        json_input = await request.json()
        obs = json_input["observation"]
        action = self.trainer.compute_action(obs)
        return {"action": int(action)}

serve.start()
ServePPOModel.deploy(checkpoint_path)

# That's it! Let's test it
for _ in range(10):
    env = gym.make("CartPole-v0")
    obs = env.reset()

    print(f"-> Sending observation {obs}")
    resp = requests.get(
        "http://localhost:8000/cartpole-ppo",
        json={"observation": obs.tolist()})
    print(f"<- Received response {resp.json()}")

