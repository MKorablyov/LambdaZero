import time, os,sys, os.path as osp
import subprocess
from LambdaZero.utils import get_external_dirs
import ray
datasets_dir, programs_dir, summaries_dir = get_external_dirs()

@ray.remote
def _sync_wandb_dir(logs_path, run):
    print("syncing run", run)
    try:
        subprocess.run(['wandb', 'sync', osp.join(logs_path, run)], check=True)
    except subprocess.CalledProcessError as e:
        print('Had issue syncing logdir %s' % run)
        print(e)


def sync_wandb(logs_path, since_sec):
    """ Synchronize all runs to the remote wandb server
    :param since_sec: time in seconds since which we want to synchronize
    :return: None
    """
    curr_time = time.time()
    # find names of all wandb runs in the directory
    runs = [l for l in os.listdir(logs_path) if l.startswith("offline-run-")]
    futures = []
    for run in runs:
        # find time when last file was changed in any of directories
        dates = [osp.getmtime(osp.join(logs_path, run, l)) for l in os.listdir(os.path.join(logs_path, run))]
        dates += [osp.getmtime(osp.join(logs_path, run, "files", l))
                  for l in os.listdir(os.path.join(logs_path, run, "files"))]
        dates += [osp.getmtime(osp.join(logs_path, run, "logs", l))
                  for l in os.listdir(os.path.join(logs_path, run, "logs"))]
        # sync if needed
        since_logged_time = int(min([curr_time - d for d in dates]))
        if since_logged_time < since_sec:
            futures.append(_sync_wandb_dir.remote(logs_path, run))
    ray.get(futures)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        since_sec = sys.argv[1]
    else:
        since_sec = 600
    ray.init()
    while True:
        sync_wandb(osp.join(summaries_dir,"wandb"),since_sec)
        time.sleep(0.5 * since_sec)
