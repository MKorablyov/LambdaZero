import gym
import numpy as np
from rdkit import Chem
from argparse import Namespace
from rdkit.Chem import QED
from lightrl.env.scores import QEDEstimator
from lightrl.env.transforms import TransformInfoDiscounted
from LambdaZero.contrib.functional import elu2


class CountBasedReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.discount_start = cfg.get("exploration_start", 101)
        self._mol_seen = dict()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done:
            next_smiles = info.get("res_molecule", None)
            if next_smiles is not None:
                if next_smiles in self._mol_seen:
                    self._mol_seen[next_smiles] += 1
                    seen = self._mol_seen[next_smiles]
                else:
                    self._mol_seen[next_smiles] = seen = 1

                scale_f = np.clip((self.discount_start-seen) / 100, 0, 1)

                if reward > 0:
                    reward *= scale_f

        return next_state, reward, done, info


class SeenInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.discount_start = cfg.get("exploration_start", 101)
        self._mol_seen = dict()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done:
            next_smiles = info.get("res_molecule", None)
            if next_smiles is not None:
                if next_smiles in self._mol_seen:
                    self._mol_seen[next_smiles] += 1
                    seen = self._mol_seen[next_smiles]
                else:
                    self._mol_seen[next_smiles] = seen = 1

                info["extra_r"] = 1 / seen

        return next_state, reward, done, info


class SeenLinearInfo(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.discount_start = cfg.get("exploration_start", 101)
        self._mol_seen = dict()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        if done:
            next_smiles = info.get("res_molecule", None)
            if next_smiles is not None:
                if next_smiles in self._mol_seen:
                    self._mol_seen[next_smiles] += 1
                    seen = self._mol_seen[next_smiles]
                else:
                    self._mol_seen[next_smiles] = seen = 1

                info["extra_r"] = np.clip((self.discount_start - seen) / 100, 0, 1)

        return next_state, reward, done, info


class ProxyCandidateReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.qed = QEDEstimator()

        self.synth = cfg.get("synth_net", None)
        self.proxy_net = cfg.get("proxy_net", None)
        self.qed_th = cfg.get("qed_th", 0.3)
        self.synth_th = cfg.get("synth_th", 4.)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        mol = self.env.molMDP.molecule
        mol_mol = mol.mol

        if info.get("res_molecule", None) is None:
            reward = -4
            return obs, reward, done, info

        reward = 0
        if done:
            qed_score = QED.qed(mol_mol)
            if qed_score >= self.qed_th:
                if self.synth is not None:
                    synth_score = self.synth([mol_mol])[0]
                    if synth_score >= self.synth_th:
                        reward = self.proxy_net(obs)[0] * -1
                        info["proxy"] = reward
                        info["score"] = reward
                        info["qed"] = qed_score
                        info["synth"] = synth_score

        return obs, reward, done, info


class ELUProxyCandidateReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.qed = QEDEstimator()

        self.synth = cfg.get("synth_net", None)
        self.proxy_net = cfg.get("proxy_net", None)
        self.qed_th = cfg.get("qed_th", 0.3)
        self.synth_th = cfg.get("synth_th", 4.)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mol = self.env.molMDP.molecule
        mol_mol = mol.mol

        if info.get("res_molecule", None) is None:
            reward = -4
            return obs, reward, done, info

        reward = 0
        if done:
            qed_score = QED.qed(mol_mol)
            if qed_score >= self.qed_th:
                if self.synth is not None:
                    synth_score = self.synth([mol_mol])[0]
                    if synth_score >= self.synth_th:
                        reward = self.proxy_net(obs)[0] * -1
                        reward = elu2([reward])[0]
                        info["proxy"] = reward
                        info["score"] = reward
                        info["qed"] = qed_score
                        info["synth"] = synth_score

        return obs, reward, done, info


class DiscountProxyCandidateReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.qed = QEDEstimator()

        self.synth = cfg.get("synth_net", None)
        self.proxy_net = cfg.get("proxy_net", None)

        self._calc_discount = TransformInfoDiscounted(score_key="proxy")

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mol = self.env.molMDP.molecule
        mol_mol = mol.mol

        reward = 0

        if info.get("res_molecule", None) is None:
            reward = 0
            return obs, reward, done, info

        if done:
            qed_score = QED.qed(mol_mol)
            synth_score = self.synth([mol_mol])[0]
            proxy = self.proxy_net(obs, norm=False)[0]

            info["proxy"] = proxy
            info["qed"] = qed_score
            info["synth"] = synth_score

            info = self._calc_discount([info])[0]

            reward = info["dscore"]
            info["score"] = reward

        return obs, reward, done, info


class MaxCandidateReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.qed = QEDEstimator()

        self.synth = cfg.get("synth_net", None)
        self.proxy_net = cfg.get("proxy_net", None)
        self.qed_th = cfg.get("qed_th", 0.3)
        self.synth_th = cfg.get("synth_th", 4.)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mol = self.env.molMDP.molecule
        mol_mol = mol.mol

        if info.get("res_molecule", None) is None:
            reward = -4
            return obs, reward, done, info

        reward = 0
        if done:
            qed_score = QED.qed(mol_mol)
            if qed_score >= self.qed_th:
                if self.synth is not None:
                    synth_score = self.synth([mol_mol])[0]
                    if synth_score >= self.synth_th:
                        reward = (qed_score / self.qed_th) * (synth_score / self.synth_th)
                        info["score"] = reward
                        info["qed"] = qed_score
                        info["synth"] = synth_score

        return obs, reward, done, info


class ActProxyCandidateReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.qed = QEDEstimator()

        self.synth = cfg.get("synth_net", None)
        self.proxy_net = cfg.get("proxy_net", None)
        self.qed_th = cfg.get("qed_th", 0.3)
        self.synth_th = cfg.get("synth_th", 4.)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mol = self.env.molMDP.molecule
        mol_mol = mol.mol

        if info.get("res_molecule", None) is None:
            reward = -4
            return obs, reward, done, info

        reward = 0
        if action == 0:
            qed_score = QED.qed(mol_mol)
            if qed_score >= self.qed_th:
                if self.synth is not None:
                    synth_score = self.synth([mol_mol])[0]
                    if synth_score >= self.synth_th:
                        reward = self.proxy_net(obs)[0] * -1
                        info["proxy"] = reward
                        info["qed"] = qed_score
                        info["synth"] = synth_score
                        info["score"] = reward

        return obs, reward, done, info


class OracleCandidateReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict

        self.qed = QEDEstimator()
        self.synth = cfg.get("synth_net", None)
        self.oracle = cfg.get("oracle", None)
        self.oracle = cfg.get("oracle", None)
        self.qed_th = cfg.get("qed_th", 0.3)
        self.synth_th = cfg.get("synth_th", 4.)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mol = self.env.unwrapped.molMDP.molecule
        mol_mol = mol.mol

        if info.get("res_molecule", None) is None:
            reward = 0
            return obs, reward, done, info

        reward = 0
        if action == 0:
            qed_score = QED.qed(mol_mol)
            if qed_score >= self.qed_th:
                synth_score = self.synth([mol_mol])[0]
                if synth_score >= self.synth_th:
                    reward = 1
                    # Send to oracle for eval
                    self.oracle([self.env.unwrapped.molMDP.dump()])

        return obs, reward, done, info


class ProxyCandidateRewardWithScoreMem(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        cfg = env.unwrapped._config  # type: dict
        self.qed = QEDEstimator()

        self.synth = cfg.get("synth_net", None)
        self.proxy_net = cfg.get("proxy_net", None)
        conns = cfg["score_mem_conn"]
        self.score_send = conns[0]
        self.score_recv = conns[1][self.env.unwrapped._env_true_proc_id]
        self.qed_th = cfg.get("qed_th", 0.3)
        self.synth_th = cfg.get("synth_th", 4.)
        self._mem = dict()

    def get_from_mem(self, smiles: str):
        # if smiles in self._mem:
        #     smi_mem = self._mem[smiles]
        # else:
        #     smi_mem = Namespace(qed=None, synth=None, proxy=None, dockscore=None)
        #     self._mem[smiles] = smi_mem
        #
        # TODO why isn't it faster ?!?!
        smi_mem = Namespace(qed=None, synth=None, proxy=None, dockscore=None)

        return smi_mem

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        mol = self.env.molMDP.molecule
        mol_mol = mol.mol

        if info.get("res_molecule", None) is None:
            reward = -4
            return obs, reward, done, info

        smi_mem = self.get_from_mem(info.get("res_molecule"))

        reward = 0
        if done:
            qed_score = smi_mem.qed = QED.qed(mol_mol) if smi_mem.qed is None else smi_mem.qed

            if qed_score >= self.qed_th:
                if self.synth is not None:
                    synth_score = smi_mem.synth = self.synth([mol_mol])[0] if smi_mem.synth is None else smi_mem.synth

                    if synth_score >= self.synth_th:

                        if smi_mem.dockscore is None:
                            self.score_send.put(
                                (0, (self.env.unwrapped._env_true_proc_id, info["mol"]))
                            )
                            req_response = self.score_recv.recv()

                            if req_response[0] is not None:
                                true_dock = req_response[1]
                                # In case we reached max failed hits on a smile
                                if true_dock is None:
                                    smi_mem.dockscore = reward = -4
                                else:
                                    smi_mem.dockscore = reward = self.proxy_net._model._norm.tfm(true_dock) * -1

                                # if req_response[0] is not None:
                                #     print(f"Nice! found predocked {req_response}")
                            else:
                                reward = smi_mem.proxy = self.proxy_net(obs)[0] * -1 if smi_mem.proxy is None else smi_mem.proxy
                            del req_response
                        else:
                            reward = smi_mem.dockscore

                        info["proxy"] = reward
                        info["score"] = reward
                        info["qed"] = qed_score
                        info["synth"] = synth_score

        return obs, reward, done, info

