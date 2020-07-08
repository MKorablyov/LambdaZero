import time, os.path as osp
from copy import deepcopy
from rdkit import Chem
import numpy as np
import ray
import LambdaZero.utils
import LambdaZero.environments

class cfg:
    datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
    num_cpus = 8

    db_name = "actor_dock"
    db_path = osp.join(datasets_dir, db_name)
    results_dir = osp.join(datasets_dir, db_name, "results")
    out_dir = osp.join(datasets_dir, db_name, "dock")

    # env parameters
    blocks_file = osp.join(datasets_dir,"fragdb/blocks_PDB_105.json")

    # MPNN parameters
    dockscore_model = osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002")

    # data parallel
    num_workers = 4


@ray.remote(num_gpus=0.1)
class Worker:
    def __init__(self):
        # initialize MDP
        self.molMDP = LambdaZero.environments.molMDP.MolMDP(blocks_file=cfg.blocks_file)
        # Initialize dockign reward prediction
        self.comp_reward = LambdaZero.environments.reward.PredDockReward(load_model=cfg.dockscore_model,
                                                                         natm_cutoff=[45, 50],
                                                                         qed_cutoff=[0.2, 0.7],
                                                                         soft_stop=False,
                                                                         exp=None,
                                                                         delta=False,
                                                                         simulation_cost=0.0,
                                                                         device="cuda")

    def sample(self):
        # do random walk
        self.molMDP.reset()
        self.comp_reward.reset()
        self.molMDP.random_walk(5)

        init_molecule = deepcopy(self.molMDP.molecule)
        # iterate over places to change
        action_values = []
        for atmidx in self.molMDP.molecule.stem_atmidxs:
            # iterate over alphabet of building blocks
            addblock_values = []
            for block_idx in range(self.molMDP.num_blocks):
                self.molMDP.add_block(block_idx=block_idx, atmidx=atmidx)
                reward = self.comp_reward(self.molMDP.molecule, env_stop=False, simulate=True, num_steps=1)[0]
                addblock_values.append(reward)
                self.molMDP.molecule = deepcopy(init_molecule)
            action_values.append(addblock_values)
        return np.asarray(action_values)



if __name__ == "__main__":

    ray.init()

    workers = [Worker.remote() for i in range(cfg.num_workers)]
    tasks = [worker.sample.remote() for worker in workers]

    task2worker = {task: worker for task, worker in zip(tasks, workers)}


    samples = []
    for i in range(20):
        done_task, tasks = ray.wait(tasks)
        done_worker = task2worker.pop(done_task[0])
        samples.append(ray.get(done_task))

        new_task = done_worker.sample.remote()
        task2worker[new_task] = done_worker
        tasks.append(new_task)


        print(samples)



        #tasks done_task[0]


       # print(ray.get(sample))



#   molMDP
#   rewars
# def sample()
#   return energies_for_molecule

# give 10 jobs per worker
# tasks = [[worker.sample() for worker in 100] for 10]

# while True:
#   # wait for 1000 jobs to complete and restart jobs
#   ray_wait_for(1000)
#   ray_restart_finished()
#   save_resuts




#def compute_action_values(molMDP):



# # iterate throuch a bunch of molecules


# molMDP().random_walk() -> mol_graph, r-groups
# choose r-group randomly
# molMDP().make_subs(r-group) -> mol_graph * 105
# MPNN(mol_graph, r_group) -> logit * 105           # actor
# MPNN(mol_graphs) -> label * 105                   # critic

#
import ray
#
# @ray.remote
# def some():
#     time.sleep(0.5)
#
# ray.init(num_cpus=6, include_webui=False, ignore_reinit_error=True)
#
# # Sleep a little to improve the accuracy of the timing measurements used below,
# # because some workers may still be starting up in the background.
# time.sleep(2.0)
#
# @ray.remote
# def f(i):
#     np.random.seed(5 + i)
#     x = np.random.uniform(0, 4)
#     time.sleep(x)
#     return i, time.time()
#
# start_time = time.time()


# # This launches 6 tasks, each of which takes a random amount of time to
# # complete.
# result_ids = [f.remote(i) for i in range(6)]
# # Get one batch of tasks. Instead of waiting for a fixed subset of tasks, we
# # should instead use the first 3 tasks that finish.
# finished_ids, remaining_ids = ray.wait(result_ids, num_returns=3)
# initial_results = ray.get(finished_ids)
#
# end_time = time.time()
# duration = end_time - start_time
#
# # Wait for the remaining tasks to complete.
# remaining_results = ray.get(remaining_ids)
#
# assert len(initial_results) == 3
# assert len(remaining_results) == 3
#
# initial_indices = [result[0] for result in initial_results]
# initial_times = [result[1] for result in initial_results]
# remaining_indices = [result[0] for result in remaining_results]
# remaining_times = [result[1] for result in remaining_results]
#
# assert set(initial_indices + remaining_indices) == set(range(6))
# assert duration < 1.5, ('The initial batch of ten tasks was retrieved in '
#                         '{} seconds. This is too slow.'.format(duration))
# assert duration > 0.8, ('The initial batch of ten tasks was retrieved in '
#                         '{} seconds. This is too slow.'.format(duration))
# # Make sure the initial results actually completed first.
# assert max(initial_times) < min(remaining_times)
# print('Success! The example took {} seconds.'.format(duration))