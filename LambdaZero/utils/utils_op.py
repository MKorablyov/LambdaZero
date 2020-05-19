import time
import os.path as osp
import configparser

def get_external_dirs():
    """Locate in the filesystem external programs/folders essensial for LambdaZero execution
    :return:
    datasets_dir (datasets),
    programs_dir (external program binaries),
    summaries_dir (where to put outpus/model summaries)
    """
    ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
    config = configparser.ConfigParser()
    config.read(osp.join(ROOT, 'setup.cfg'))

    if config["dir"]["datasets"].startswith("/"):
        datasets_dir = config["dir"]["datasets"]
    else:
        datasets_dir = osp.join(ROOT, config["dir"]["datasets"])
    if config["dir"]["programs"].startswith("/"):
        programs_dir = config["dir"]["programs"]
    else:
        programs_dir = osp.join(ROOT, config["dir"]["programs"])
    if config["dir"]["summaries"].startswith("/"):
        summaries_dir = config["dir"]["summaries"]
    else:
        summaries_dir = osp.join(ROOT, config["dir"]["summaries"])

    return datasets_dir, programs_dir, summaries_dir


def dock_metrics(info):
    """
    Report custom metrics for each episode in RayRllib
    :param info: episode info
    :return:
    """
    env_info = list(info["episode"]._agent_to_last_info.values())[0]
    episode = info["episode"]
    episode.custom_metrics["reward"] = env_info["reward"]
    episode.custom_metrics["QED"] = env_info["QED"]
    episode.custom_metrics["discounted_reward"] = env_info["discounted_reward"]