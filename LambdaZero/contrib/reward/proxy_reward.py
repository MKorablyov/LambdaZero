import time
from LambdaZero.contrib.proxy import Actor


class ProxyReward:
    def __init__(self, scoreProxy, Proc, proc_config, actor_sync_freq, **kwargs):
        self.actor = Actor(scoreProxy, Proc, proc_config, actor_sync_freq)

    def reset(self):
        return None

    def __call__(self, molecule, obs, agent_stop, env_stop, num_steps):
        synth_score = 0.5
        qed = 0.9

        dock_score = self.actor([{"molecule":molecule, "mol_graph":obs["mol_graph"]}], [qed * synth_score])[0]
        scores = {"dock_score":dock_score, "synth_score": synth_score, "qed":0.9}

        return synth_score * dock_score * qed, scores

