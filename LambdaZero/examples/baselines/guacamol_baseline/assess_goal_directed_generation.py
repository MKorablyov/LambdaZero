import datetime
import json
import logging
from collections import OrderedDict
from typing import List, Any, Dict

import guacamol
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark, GoalDirectedBenchmarkResult
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.assess_goal_directed_generation import _evaluate_goal_directed_benchmarks

# from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.utils.data import get_time_string

from LambdaZero.examples.baselines.guacamol_baseline.scoring_functions import Oracle_wrapper, Proxy_wrapper
from LambdaZero.examples.bayesian_models.rl.train_bayesian_reward import config
from LambdaZero.environments.reward import BayesianRewardActor
from LambdaZero.environments.reward import PredDockBayesianReward_v1
from LambdaZero.environments.block_mol_v3 import DEFAULT_CONFIG as block_mol_v3_config

import ray
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def assess_goal_directed_generation(goal_directed_molecule_generator: GoalDirectedGenerator,
                                    json_output_file='output_goal_directed.json',
                                    benchmark_version='v1') -> None:
    """
    Assesses a distribution-matching model for de novo molecule design.

    Args:
        goal_directed_molecule_generator: Model to evaluate
        json_output_file: Name of the file where to save the results in JSON format
        benchmark_version: which benchmark suite to execute
    """
    logger.info(f'Benchmarking goal-directed molecule generation, version {benchmark_version}')
    benchmarks = goal_directed_benchmark_suite(version_name=benchmark_version)

    results = _evaluate_goal_directed_benchmarks(
        goal_directed_molecule_generator=goal_directed_molecule_generator,
        benchmarks=benchmarks)

    benchmark_results: Dict[str, Any] = OrderedDict()
    benchmark_results['guacamol_version'] = guacamol.__version__
    benchmark_results['benchmark_suite_version'] = benchmark_version
    benchmark_results['timestamp'] = get_time_string()
    benchmark_results['results'] = [vars(result) for result in results]

    logger.info(f'Save results to file {json_output_file}')
    with open(json_output_file, 'wt') as f:
        f.write(json.dumps(benchmark_results, indent=4))

def goal_directed_benchmark_suite(version_name: str) -> List[GoalDirectedBenchmark]:
    if version_name == 'v1':
        return oracle()
    if version_name == 'v2':
        ray.init(object_store_memory=config["object_store_memory"], _memory=config["memory"])
        reward_learner = BayesianRewardActor.options(
            num_cpus=config["reward_learner_config"]['reward_actor_cpus'],
            num_gpus=config["reward_learner_config"]['reward_actor_gpus']). \
            remote(config["reward_learner_config"], config["use_dock"],
                   config["rllib_config"]['env_config']['reward_config']['binding_model'], config["pretrained_model"])

        return proxy(reward_learner)

def oracle() -> List[GoalDirectedBenchmark]:
    return [
        docking_scoring_function(),
    ]

def proxy(reward_learner) -> List[GoalDirectedBenchmark]:
    return [
        proxy_scoring_function(reward_learner),
    ]

def docking_scoring_function() -> GoalDirectedBenchmark:
    benchmark_name = f'dockscore_seh_protein'
    objective = Oracle_wrapper()

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)

def proxy_scoring_function(reward_learner) -> GoalDirectedBenchmark:
    benchmark_name = f'proxy_dockscore_seh_protein'
    block_mol_v3_config["reward_config"]["dense_rewards"] = True
    block_mol_v3_config['reward_config']['reward_learner'] = reward_learner
    block_mol_v3_config['reward_config']['regressor'] = config['reward_learner_config']['regressor']
    block_mol_v3_config['reward_config']['regressor_config'] = config['reward_learner_config']['regressor_config']
    block_mol_v3_config['reward_config']['kappa'] = config['reward_learner_config']['kappa']
    block_mol_v3_config['reward_config']['sync_freq'] = config['reward_learner_config']['sync_freq']

    objective = Proxy_wrapper(PredDockBayesianReward_v1, block_mol_v3_config)

    specification = uniform_specification(1, 10, 100)

    return GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)