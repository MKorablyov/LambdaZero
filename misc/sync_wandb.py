import time, os, os.path as osp
import subprocess
from LambdaZero.utils import get_external_dirs
datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def sync_wandb(logs_path, since_time):
    """ Synchronize all runs to the remote wandb server
    :param since_time: time in seconds since which we want to synchronize
    :return: None
    """
    curr_time = time.time()
    # find names of all wandb runs in the directory
    runs = [l for l in os.listdir(logs_path) if l.startswith("offline-run-")]
    for run in runs:
        # find time when last file was changed in any of directories
        dates = [osp.getmtime(osp.join(logs_path, run, l)) for l in os.listdir(os.path.join(logs_path, run))]
        dates += [osp.getmtime(osp.join(logs_path, run, "files", l))
                  for l in os.listdir(os.path.join(logs_path, run, "files"))]
        dates += [osp.getmtime(osp.join(logs_path, run, "logs", l))
                  for l in os.listdir(os.path.join(logs_path, run, "logs"))]
        # sync if needed
        since_logged_time = int(min([curr_time - d for d in dates]))  #
        if since_logged_time < since_time:
            try:
                subprocess.run(['wandb', 'sync', osp.join(logs_path, run)], check=True)
            except subprocess.CalledProcessError as e:
                print('Had issue syncing logdir %s' % run)
                print(e)


if __name__ == "__main__":
    while True:
        sync_wandb(osp.join(summaries_dir,"wandb"),1195)
        time.sleep(600)


