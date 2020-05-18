import os
import os.path as osp
import tempfile

import ray
import pandas as pd

from LambdaZero.examples.AlphaZero.config import mol_blocks_v4_config
from LambdaZero.chem import Dock_smi

DATASET_PATH = "/fsx/datasets/zinc15_2D/splits/"
RESULTS_PATH = "/fsx/results/docking/"


class Empty:
    def __init__(self, **kwargs):
        pass

    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        return None, {}


class cfg:
    # temp
    ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
    datasets_dir = osp.join(ROOT, "Datasets")
    programs_dir = osp.join(ROOT, "Programs")

    # change
    db_name = "actor_dock"
    results_dir = osp.join(datasets_dir, db_name, "results")
    dock_dir = osp.join(datasets_dir, db_name, "dock")

    # env
    env_config = mol_blocks_v4_config()["env_config"]
    env_config["random_steps"] = 3
    env_config["reward"] = Empty
    env_config["obs"] = Empty

    # docking parameters
    dock6_dir = osp.join(programs_dir, "dock6")
    chimera_dir = osp.join(programs_dir, "chimera")
    docksetup_dir = osp.join(datasets_dir, "brutal_dock/mpro_6lze/docksetup")


# for 5 123 456 molecules
# Results are organized in a hierarchy (51/23/456.res)
# if 51.done exists then all the molecules under 51/ are done.
# if 51/23.done exists then all the molecules under 51/23/ are done.


def _find_top(init):
    i = init
    while osp.exists(osp.join(DATASET_PATH, "{0:#02}.done".format(i))):
        i = i + 1
        if i == 100:
            # There is no higer level so just give up
            return None
        if not osp.exists(osp.join(DATASET_PATH, 'mol{0:#02}0000'.format(i))):
            return None
    ip = "{0:#02}".format(i)
    os.makedirs(osp.join(RESULTS_PATH, ip), exist_ok=True)
    return i, ip


def _find_mid(ip, init):
    j = init

    while osp.exists(osp.join(RESULTS_PATH, ip, "{0:#02}.done".format(j))):
        j = j + 1
        if j == 100:
            return 'next'
        if not osp.exists(osp.join(DATASET_PATH, 'mol{0}{1:#02}00'.format(ip, j))):
            return None
    jp = "{0:#02}".format(j)
    os.makedirs(osp.join(RESULTS_PATH, ip, jp), exist_ok=True)
    return j, jp


def _find_bottom(ip, jp, init):
    k = init
    while osp.exists(osp.join(RESULTS_PATH, ip, jp, "{0:#02}.parquet".format(k))):
        k = k + 1
        if k == 100:
            return 'next'
        if not osp.exists(osp.join(DATASET_PATH, 'mol{0}{1}{2:#02}'.format(ip, jp, k))):
            return None
    return k


def molpath(i, j, k):
    return osp.join(DATASET_PATH, "mol{0:#02}{1:#02}{2:#02}".format(i, j, k))


def outpath(i, j, k):
    return osp.join(RESULTS_PATH, "{0:#02}".format(i), "{0:#02}".format(j),
                    "{0:#02}.parquet".format(k))


def find_next_batch(init_i=0, init_j=0):
    i = init_i
    j = init_j
    k = 0

    while True:
        res = _find_top(i)
        if res is None:
            return None
        i, ip = res

        res = _find_mid(ip, j)
        if res is None:
            return None
        if res == 'next':
            # Create the .done file
            if j == 0:
                open(osp.join(RESULTS_PATH, ip + ".done"), 'a').close()
            i += 1
            if i == 100:
                return None
            j = 0
            k = 0
            continue
        j, jp = res

        res = _find_bottom(ip, jp, k)
        if res is None:
            return None
        if res == 'next':
            # Create the .done file
            if k == 0:
                open(osp.join(RESULTS_PATH, ip, jp + ".done"), 'a').close()
            j += 1
            if j == 100:
                i += 1
                if i == 100:
                    return None
                j = 0
            k = 0
            continue
        k = res
        return i, j, k


@ray.remote(num_cpus=1)
def do_docking(i, j, k, results_dir):
    real_home = os.getenv('REAL_HOME', os.getenv('HOME'))
    os.environ['REAL_HOME'] = real_home
    os.environ['HOME'] = real_home + "/homes/{0}/{1}/{2}/".format(i, j, k)
    os.makedirs(os.environ['HOME'], exist_ok=True)
    workpath = "/tmp/docking/{0}/{1}/{2}/".format(i, j, k)
    os.makedirs(workpath, exist_ok=True)
    dock_smi = Dock_smi(outpath=workpath,
                        chimera_dir=cfg.chimera_dir,
                        dock6_dir=cfg.dock6_dir,
                        docksetup_dir=cfg.docksetup_dir,
                        #gas_charge=True,
                        trustme=True)

    results = []
    # Dock all the molecules in the file
    with open(molpath(i, j, k), 'r') as f:
        for n, line in enumerate(f):
            smi, zinc_id, _, mwt, logP, _, _, tranche, *features = line.rstrip().split()
            idx = int(zinc_id[4:])
            reactivity = tranche[2]
            features = features or None
            try:
                name, gridscore, coord = dock_smi.dock(smi, mol_name=str(n))
                coord = coord.tolist()
            except:
                print("Failed to dock for: {0} in {1}/{2}/{3}".format(smi, i, j, k))
                coord = None
                gridscore = None
            results.append(pd.DataFrame({"smi": [smi],
                                         "gridscore": [gridscore],
                                         "coord": [coord],
                                         "mwt": [mwt],
                                         "logP": [logP],
                                         "zinc_id": [idx],
                                         "reactivity": [reactivity],
                                         "features": [features]}))
    output = pd.concat(results, ignore_index=True)

    # This dance is to avoid partial files in the final output
    output_path = outpath(i, j, k)
    ofd, output_tmp = tempfile.mkstemp(dir=osp.dirname(output_path),
                                       suffix='.tmp')
    os.close(ofd)
    output.to_parquet(output_tmp, engine="fastparquet", compression=None)
    os.rename(output_tmp, output_path)


@ray.remote(resources={'aws-machine': 1}, num_cpus=1)
def distribute_mols(init_i=0, init_j=0):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    i = init_i
    j = init_j
    wrapped = False
    while True:
        res = find_next_batch(i, j)
        if res is None:
            if wrapped:
                return 'done'
            # Wrap around the end and try to find more work.
            wrapped = True
            i = 0
            j = 0
            continue
        i, j, k = res
        job_ids = [do_docking.remote(i, j, p, RESULTS_PATH)
                   for p in range(k, 100)
                   if osp.exists(molpath(i, j, p))]

        # Wait for the jobs to finish
        ray.get(job_ids)

        # Mark the subdir as done
        open(osp.join(RESULTS_PATH, "{0:#02}".format(i), "{0:#02}.done".format(j)), 'a').close()


if __name__ == '__main__':
    ray.init(address='auto')
    num_dispatchers = 176  # To have 17000~ per machine
    total_data = 5997 # Approximative, in thousands
    parts_per_dispatch = total_data // num_dispatchers
    dispatchers = [distribute_mols.remote(
        init_i=(p * parts_per_dispatch) // 100,
        init_j=(p * parts_per_dispatch) % 100) for p in range(num_dispatchers)]

    # Wait for the jobs to finish
    ray.get(dispatchers)
