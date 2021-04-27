import time
from rdkit import Chem
from rdkit.Chem import Draw

def log_episode_info(info):
    env_info = list(info["episode"]._agent_to_last_info.values())[0]
    episode = info["episode"]
    #episode.user_data["molecule"] = env_info["molecule"]
    for key, value in env_info["log_vals"].items():
        episode.custom_metrics[key] = value


class TrialNameCreator:
    def __init__(self, config_name):
        self.config_name = config_name

    def __call__(self, trial):
        return "{}_{}_{}".format(self.config_name, trial.trainable_name, trial.trial_id)


