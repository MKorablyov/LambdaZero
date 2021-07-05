import cv2
import numpy as np
from multiprocessing import Process, Pipe, Queue
import multiprocessing
import copy
from argparse import Namespace

from LambdaZero.contrib.oracle.oracle import DockVina_smi_db
from LambdaZero.contrib.oracle.oracle import config_DockingOracle_v1 as oracle_config
from LambdaZero.contrib.oracle import DockingOracle


class FakeRemoteLog:
    def __init__(self):
        self._last_log = None
        self.log = Namespace(remote=self.remote)

    def remote(self, log):
        self._last_log = log


class DockingEstimatorThread(DockVina_smi_db):
    """ Docking estimator separate for each env. """
    def __init__(self, dockVina_config):
        DockVina_smi_db.__init__(self, **dockVina_config)

    def eval(self, smiles, **kwargs):
        try:
            mol_name, dockscore, coord = self.dock(smiles, **kwargs)
        except Exception as e:
            dockscore = None
        return dockscore


def compute_dock(conn, conn_send):
    cfg = copy.deepcopy(oracle_config["dockVina_config"])
    cfg["query_predocked"] = False
    dockestm = DockingEstimatorThread(cfg)

    while True:
        _id, smiles, data = conn.get()
        ret = dockestm.eval(smiles, mol_data=data)
        conn_send.put((_id, smiles, ret))


class InterogateOracle:
    """
        Interogate global thread for looking up predocked scores. (1 db loaded)
        If not found use local thread to calculate oracle (+update db)
    """

    def __init__(self, num_workers=4, send_updates_conn=None):
        fake_logger = FakeRemoteLog()
        oracle_config["logger"] = fake_logger
        oracle_config["num_threads"] = 0
        dock_oracle = DockingOracle(**oracle_config)

        self._send_updates_conn = send_updates_conn
        self._dock_db = dock_oracle
        self._dock_db.set_fix_none(False)
        self._dock_db.set_norm_data(False)

        self.remote_send = remote_send = Queue()
        self.remote_recv = remote_recv = Queue()

        self.processes = []
        for _ in range(num_workers):
            p = Process(
                target=compute_dock,
                args=(remote_send, remote_recv))
            p.daemon = True
            p.start()

            self.processes.append(p)

        self._oracle_req = dict()
        self._ready_req = dict()
        self._max_send_pending_que = num_workers * 2
        self._pending_que = []

        # self.mean = oracle_config["mean"]
        # self.std = oracle_config["std"]
        # self.act_y = oracle_config["act_y"]

    def __call__(self, x):
        _id = np.random.randint(10000)

        # Send only
        smi_data = [{"smiles": i["smiles"]} for i in x]
        res, tres = self._dock_db(smi_data, ret_true_dockscore=True)

        for i, (smi, data, dockscore, true_dock) in enumerate(zip(smi_data, x, res, tres)):
            if dockscore is None and smi["smiles"] not in self._oracle_req:
                self._oracle_req[smi["smiles"]] = True
                self.remote_send.put((_id, smi["smiles"], data))
            elif dockscore is not None:
                # In case we reached max failed hits on a smile we should put None
                self._ready_req[smi["smiles"]] = true_dock

    def update_pending_que(self, requests=[]):
        self.empty_recv_que()

        if len(requests) > 0:
            self._pending_que = requests

        while len(self._oracle_req) < self._max_send_pending_que and len(self._pending_que) > 0:
            new_req = self._pending_que.pop(0)

            # Double check we do not have score already
            dockscore, true_dock = self.check_db([new_req])[0]
            if dockscore is None and new_req["smiles"] not in self._oracle_req:
                _id = np.random.randint(10000)

                self._oracle_req[new_req["smiles"]] = True
                self.remote_send.put((_id, new_req["smiles"], new_req))

    def check_db(self, x):
        # Send only
        smi_data = [{"smiles": i["smiles"]} for i in x]
        res, tres = self._dock_db(smi_data, ret_true_dockscore=True)
        return list(zip(res, tres))

    def empty_recv_que(self):
        while not self.remote_recv.empty():
            _id, smiles, ret = self.remote_recv.get()
            self._dock_db.update_predocked(smiles, ret)
            self._oracle_req.pop(smiles)

    def get_req(self, smiles=None):
        if smiles is not None:
            raise NotImplementedError

        # Return all req
        req = self._ready_req
        self._ready_req = dict()

        while len(self._oracle_req) > 0:
            _id, smiles, ret = self.remote_recv.get()

            if self._send_updates_conn is not None:
                self._send_updates_conn.put((smiles, ret))

            self._dock_db.update_predocked(smiles, ret)
            req[smiles] = ret
            self._oracle_req.pop(smiles)
        return req


if __name__ == "__main__":
    import gym
    from LambdaZero.examples.lightrl.env.reward import DummyReward
    import numpy as np

    oracle = InterogateOracle()

    env = gym.make("BlockMolEnvGraph-v1", config={
        "reward": DummyReward,
        "random_steps": 4,
        "max_steps": 5,
        "allow_removal": True,
        "env_seed": 15}, proc_id=0)

    obs = env.reset()
    act = np.random.choice(np.where(obs["action_mask"])[0])
    obs, r, done, info = env.step(act)
    req_dock = [env.unwrapped.molMDP.dump()]
    oracle(req_dock)
    oracle(req_dock)
    oracle(req_dock)
    req_dock[0]["smiles"] = "sdasdas"
    oracle(req_dock)
    oracle(req_dock)
    oracle(req_dock)

    # pre docked
    env = gym.make("BlockMolEnvGraph-v1", config={
        "reward": DummyReward,
        "random_steps": 3,
        "max_steps": 1,
        "allow_removal": True,
        "env_seed": [13]}, proc_id=0)

    obs = env.reset()

    for i in range(10):
        act = np.random.choice(np.where(obs["action_mask"])[0])
        obs, r, done, info = env.step(act)

        req_dock = [env.unwrapped.molMDP.dump()]
        oracle(req_dock)

        if done:
            obs = env.reset()

    ret_req = oracle.get_req()
    print(ret_req)
