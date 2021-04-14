import numpy as np
import json
import sys
import os.path as osp
import os
import wandb
import logging

from typing import List, Optional
from scipy import special

from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.assess_goal_directed_generation import assess_goal_directed_generation
from guacamol.utils.chemistry import canonicalize_list, canonicalize
from guacamol.utils.helpers import setup_default_logger
import LambdaZero.examples.baselines.guacamol.config as config
from LambdaZero.examples.baselines.hiv.apollo_scoringfunction import Apollo_Scoringfunction
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

# from LambdaZero.examples.baselines.Apollo1060.pipeliner_light.pipelines import ClassicPipe
# from LambdaZero.examples.baselines.Apollo1060.pipeliner_light.smol import SMol
from guacamol.scoring_function import ScoringFunction, BatchScoringFunction

os.environ["WANDB_MODE"] = "dryrun"
with open(osp.join(summaries_dir, "wandb_key"), 'r') as file:
    key = file.read().replace('\n', '')
os.environ["WANDB_API_KEY"] = key
os.environ["WANDB_DIR"] = summaries_dir

class OptResult:
    def __init__(self, smiles=None, score=0) -> None:
        self.smiles = smiles
        self.score = score

class BoltzmannDirectedGenerator(GoalDirectedGenerator):
    def __init__(self, config):
        # config["env_config"]["reward_config"]["device"] = "cpu"
        self.env = config["env"](config["env_config"])

        self.temperature = config["temperature"]
        self.steps = config["steps"]
        self.num_samples = config["num_samples"]

    def enumerate_actions(self, scoring_function):
        state = self.env.get_state()
        actions = np.where(self.obs["action_mask"])[0]
        disc_rewards = []
        scores = []
        for i, a in enumerate(actions):
            # temp_obs, reward, _, info = self.env.step(a)
            temp_obs, disc_reward, _, info = self.env.step(a)
            if info["molecule"] is not None:
                score = scoring_function.score(info["molecule"]) # self.env.molMDP.molecule.mol) #
                reward = disc_reward * score
            else:
                score = 0
                reward = 0
            scores.append(score)
            disc_rewards.append(reward)
            self.env.set_state(state)
        probs = special.softmax(np.divide(disc_rewards, self.temperature))
        a = np.random.choice(actions.shape[0], p=probs)
        return actions[a], disc_rewards[a], scores[a]

    def generate_optimized_molecules(self, scoring_function: ScoringFunction, number_molecules: int,
                                     starting_population: Optional[List[str]] = None) -> List[str]:
        wandb.init(project="retrognn_benchmarks", reinit=True)
        wandb.run.name = str(config_name)
        top_mols = [OptResult() for x in range(number_molecules)]
        for i in range(self.num_samples):
            # make initial observation
            self.obs = self.env.reset()
            for j in range(self.steps):
                action, disc_reward, score = self.enumerate_actions(scoring_function)
                self.obs, _, _, info = self.env.step(action)
                smiles = info["molecule"]
                if smiles is not None:
                    reward = score # scoring_function.score(smiles)  # technically returns a list
                else:
                    reward = 0
                if reward > top_mols[-1].score:
                    del top_mols[-1]
                    top_mols.append(OptResult(smiles, reward))
                    top_mols.sort(key=lambda x: x.score, reverse=True)
                # print(j)
                wandb.log({"step": (i+1)*(j+1), "reward": reward})
            print([m.smiles for m in top_mols], [m.score for m in top_mols])
        samples = [m.smiles for m in top_mols]
        samples = canonicalize_list(samples)
        return samples

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        config_name = sys.argv[1]
    else:
        config_name = "boltzmann_config_000"
    config = getattr(config, config_name)

    setup_default_logger()
    optimizer = BoltzmannDirectedGenerator(config["boltzmann_config"])

    scoring_functions = config["scoring_functions"]
    json_file_path = osp.join(summaries_dir, str(config)+'_goal_directed_results.json')
    if scoring_functions == 'v1' or scoring_functions == 'v2':
        assess_goal_directed_generation(optimizer, json_output_file=json_file_path, benchmark_version=scoring_functions)
    elif scoring_functions == 'apollo':
        scoring_functions = Apollo_Scoringfunction(config["apollo_pipe"])
        molecules = optimizer.generate_optimized_molecules(scoring_functions, number_molecules=50, starting_population=[])
        scores = scoring_functions.score_list(molecules)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        logger.info(f'  Molecules: {molecules:.6f}')
        logger.info(f'  Scores: {scores:.6f}')
        logger.info(f'Save results to file {json_file_path}')
        with open(json_file_path, 'wt') as f:
            f.write(json.dumps(molecules, indent=4))
            f.write(json.dumps(scores, indent=4))
        logger.info('Finished execution of the benchmarks')