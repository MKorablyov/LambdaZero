import numpy as np
import LambdaZero


class PredockedReward:
    """ Dummy reward """

    def __init__(self, **kwargs):
        datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
        self.mol_reward = mol_reward = dict()

        data = np.load(f"{datasets_dir}/eval_env_data.npy", allow_pickle=True).item()  # type: dict

        for smi, smi_data in data.items():
            next = smi_data["next"][["next_smiles", "true_r"]].set_index("next_smiles")
            mol_reward.update(next.to_dict()["true_r"])

    def reset(self, previous_reward=0.0):
        self.previous_reward = 0.0
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        # r = self.mol_reward.get(molecule.smiles, -50)
        r = self.mol_reward[molecule.smiles]
        info = {"reward": r,
                "discounted_reward": r,
                "discount": 1.0}

        return r, info


class DummyReward:
    """ Dummy reward """

    def __init__(self, **kwargs):
        pass

    def reset(self, previous_reward=0.0):
        self.previous_reward = 0.0
        return None

    def __call__(self, molecule, agent_stop, env_stop, num_steps):
        r = 0
        info = {"reward": 0,
                "discounted_reward": 0,
                "discount": 1.0}

        return r, info
