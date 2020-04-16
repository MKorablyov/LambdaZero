import os
import os.path as osp
import ray

DATASET_PATH = "/fsx/datasets/zinc15_2D/splits/"
RESULTS_PATH = "/fsx/results/docking/"

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
        if not osp.exists(osp.join(DATASET_PATH, 'mol{0:#02}00000'.format(i))):
            return None
    ip = "{0:#02}".format(i)
    os.makedirs(ops.join(RESULTS_PATH, ip), exist_ok=True)
    return i, ip


def _find_mid(ip, init):
    j = init

    while osp.exists(osp.join(RESULTS_PATH, ip, "{0:#02}.done".format(j))):
        j = j + 1
        if j == 100:
            return 'next'
        if not osp.exists(osp.join(DATASET_PATH, 'mol{0}{1:#02}000'.format(ip, j))):
            return None
    jp = "{0:#02}".format(j)
    os.makedirs(ops.join(RESULTS_PATH, ip, jp), exist_ok=True)
    return j, jp


def _find_bottom(ip, jp, init)
    k = init
    while osp.exists(osp.join(RESULTS_PATH, ip, jp "{0:#02}.gz".format(k))):
        k = k + 1
        if k == 1000:
            return 'next'
        if not osp.exists(osp.join(DATASET_PATH, 'mol{0}{1}{2:#03}'.format(ip, jp, k))):
            return None
    return k


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


@ray.remote(resources={'aws_machine': 1}, num_cpus=1)
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
                   for p in range(k, 1000)]
        ray.get(job_ids)
        # Mark the subdir as done
        open(osp.join(RESULTS_PATH, "{0:#02}".format(i), "{0:#02}.done".format(j)), 'a').close()
