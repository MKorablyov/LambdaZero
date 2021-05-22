import datetime
import json
import logging
from collections import OrderedDict
from typing import List, Any, Dict

import LambdaZero.utils
from LambdaZero.examples.baselines.guacamol_baseline.scoring_functions import Oracle_wrapper, Proxy_wrapper
from LambdaZero.examples.baselines.guacamol_baseline.guacamol_baseline_config import default_tune_config as config
# from LambdaZero.examples.bayesian_models.rl.train_bayesian_reward import config

import guacamol
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark, GoalDirectedBenchmarkResult
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.assess_goal_directed_generation import _evaluate_goal_directed_benchmarks

# from guacamol.benchmark_suites import goal_directed_benchmark_suite
from guacamol.utils.data import get_time_string


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

    if benchmark_version == 'oracle':
        benchmarks = docking_scoring_function()
    elif benchmark_version == 'proxy':
        benchmarks = proxy_scoring_function(config)
    else:
        raise Exception('scoring function not implemented')

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

def docking_scoring_function() -> List[GoalDirectedBenchmark]:
    benchmark_name = f'dockscore_seh_protein'
    objective = Oracle_wrapper()

    specification = uniform_specification(1, 10, 100)

    return [GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)]

def proxy_scoring_function(config) -> List[GoalDirectedBenchmark]:
    benchmark_name = f'proxy_dockscore_seh_protein'
    objective = Proxy_wrapper(config)
    specification = uniform_specification(1, 10, 100)
    return [GoalDirectedBenchmark(name=benchmark_name,
                                 objective=objective,
                                 contribution_specification=specification)]

