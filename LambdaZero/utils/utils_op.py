import time
import os.path as osp
import configparser

def get_external_dirs():
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