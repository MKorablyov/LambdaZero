import os
import os.path as osp
import tempfile

import ray
import pandas as pd


RESULTS_PATH = "/fsx/results/docking/"

# for 5 123 456 molecules
# Results are organized in a hierarchy (51/23/456.res)
# if 51.done exists then all the molecules under 51/ are done.
# if 51/23.done exists then all the molecules under 51/23/ are done.

def outpath(i, j, k):
    return osp.join(RESULTS_PATH, "{0:#02}".format(i), "{0:#02}".format(j),
                    "{0:#02}.parquet".format(k))

@ray.remote(num_cpus=1)
def check(i):
    ok = 0
    fail = 0
    nothere = 0
    for j in range(100):
        for k in range(100):
            f = outpath(i, j, k)
            if osp.exists(f):
                try:
                    df = pd.read_parquet(f, engine='fastparquet')
                    assert len(df) == 10
                    ok += 1
                except:
                    fail += 1
                    os.remove(f)
            else:
                nothere += 1
    return ok, nothere, fail

if __name__ == '__main__':
    ray.init(address='auto')

    jobs = [check.remote(i) for i in range(60)]

    res = ray.get(jobs)

    ok = sum(r[0] for r in res)
    nothere = sum(r[1] for r in res)
    fail = sum(r[2] for r in res)

    print("OK:", ok)
    print("Nothere:", nothere)
    print("Fail:", fail)
