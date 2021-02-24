from pathlib import Path
import subprocess
import os

HAVE_SYNCED_FNAME = '.have_synced'


def _have_synced_file_path(dir_name):
    dir_path = dir_name if isinstance(dir_name, Path) else Path(dir_name)
    return dir_path / HAVE_SYNCED_FNAME


def _get_child_dirs(dir_name):
    dir_path = dir_name if isinstance(dir_name, Path) else Path(dir_name)
    return filter(lambda x: x.is_dir(), dir_path.iterdir())


def should_sync_dir(dir_path):
    wandb_path = dir_path / 'wandb'
    have_synced_path = _have_synced_file_path(dir_path)
    if not wandb_path.exists():
        return False

    if not have_synced_path.exists():
        return True

    # Check if there's any new run dirs
    # Add -1 to the list in case there's no run directories so that
    # max does not throw
    run_times = [-1] + list(map(lambda x: x.stat().st_mtime, _get_child_dirs(wandb_path)))
    return max(run_times) > have_synced_path.stat().st_mtime


def _get_run_dirs(dir_path):
    wandb_dir = dir_path / 'wandb'
    have_synced_path = _have_synced_file_path(dir_path)
    child_dirs = _get_child_dirs(wandb_dir)

    if not have_synced_path.exists():
        return child_dirs

    have_synced_mod_time = have_synced_path.stat().st_mtime
    return filter(lambda x: x.stat().st_mtime > have_synced_mod_time, child_dirs)


def sync_dir(dir_path):
    print('Beginning sync for path %s\n' % dir_path)

    for run_dir in _get_run_dirs(dir_path):
        try:
            subprocess.run(['wandb', 'sync', run_dir], check=True)
        except subprocess.CalledProcessError as e:
            print('Had issue syncing logdir %s' % run_dir)
            print(e)


    _have_synced_file_path(dir_path).touch()

    print('\nFinished sync for path %s\n' % dir_path)


def sync_all(logdir):
    dirs_to_sync = filter(lambda x: should_sync_dir(x), _get_child_dirs(logdir))
    for dir_to_sync in dirs_to_sync:
        sync_dir(dir_to_sync)


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ray_logs_dir = os.path.join(dir_path, 'RayLogs')

    sync_all(ray_logs_dir)

