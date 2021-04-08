import time, os,sys, os.path as osp
import subprocess
import threading
from LambdaZero.utils import get_external_dirs

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


def _sync_wandb_dir(logs_path, run):
    try:
        subprocess.run(['wandb', 'sync', osp.join(logs_path, run)], check=True)
    except subprocess.CalledProcessError as e:
        print('syncing logdir %s' % run)
        print(e)


def sync_wandb(logs_path, since_sec):
    """ Synchronize all runs to the remote wandb server
    :param since_sec: time in seconds since which we want to synchronize
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
        since_logged_time = int(min([curr_time - d for d in dates]))
        if since_logged_time < since_sec:
            print("syncing", logs_path)
            threading.Thread(target=_sync_wandb_dir, args=(logs_path, run)).start()
            time.sleep(10) # sleep 1 second to prevent being blocked by google_api wandb is using


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        since_sec = int(sys.argv[1])
    else:
        since_sec = 600
    #while True: in practice it seems syncing recurrently would make wandb to
    sync_wandb(osp.join(summaries_dir,"rlbo8"),since_sec)
    # todo: in order to sync recurrently I think the right way would
    print("synced all runs sleeping", 0.5 * since_sec)
    #time.sleep(0.5 * since_sec)
