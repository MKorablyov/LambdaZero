import cv2
from typing import List, Any
import numpy as np
import time
from argparse import Namespace
from torch.multiprocessing import Process, Queue
import plotly.express as px
import plotly.graph_objects as go

from LambdaZero.examples.lightrl.utils.utils import LogTopStats


def calc_metrics(key: str, values: List[Any]) -> dict:
    if len(values) > 0:
        return {
            f"{key}_mean": np.mean(values),
            f"{key}_min": np.min(values),
            f"{key}_max": np.max(values),
            f"{key}_median": np.median(values),
            f"{key}_cnt": len(values)
        }
    return {f"{key}_cnt": len(values)}


class LogMetrics:
    def __init__(self, metrics: List[str]):
        self.metrics = {
            x: [] for x in metrics
        }
        self.last_get = {x: 0 for x in metrics}

    def add_metrics(self, metrics: List[str]):
        self.metrics.update({
            x: [] for x in metrics
        })
        self.last_get.update({x: 0 for x in metrics})

    def update(self, names: List[str], values: List[Any]):
        for x_name, x_val in zip(names, values):
            self.metrics[x_name].append(x_val)

    def get_values(self, names: List[str], dict_format=True):
        if dict_format:
            return_info = dict({self.metrics[x] for x in names})
        else:
            return_info = [self.metrics[x] for x in names]

        return return_info

    def get_values_flatt(self, name: str, new=True) -> List[Any]:
        if new:
            values = self.metrics[name][self.last_get[name]:]
            self.last_get[name] = len(self.metrics[name])
        else:
            values = self.metrics[name]

        if len(values) > 0:
            if isinstance(values[0], np.ndarray):
                values = sum([x.flatten().tolist() for x in values], [])
            elif isinstance(values[0], list):
                values = sum(values, [])

        return values

    def get_stats(self, names: List[str], new=True):
        stats = dict()
        for x in names:
            values = self.get_values_flatt(x, new=new)
            if len(values) == 0:
                continue
            stats.update(calc_metrics(x, values))

        return stats


class TimeStats:
    def __init__(self):
        self._buffer = []
        self._max = 100
        self._st_ts = time.time()

    def start(self):
        self._st_ts = time.time()

    def end(self):
        fin = time.time() - self._st_ts
        self._buffer.append(fin)
        if len(self._buffer) > self._max:
            self._buffer.pop(0)
        return np.mean(self._buffer)

    def stats(self):
        return np.mean(self._buffer)


def log_stats_remote(conn_recv, conn_send, log_topk, do_dockscore):
    if do_dockscore:
        score_keys = ("proxy", "qed", "synth", "dockscore", "dscore")
    else:
        score_keys = ("proxy", "qed", "synth")

    log_stats = LogTopStats(topk=log_topk, unique_key="smiles", score_keys=score_keys)

    while True:
        cmd, recv = conn_recv.get()
        if cmd == 0:
            log_stats.collect(*recv)
        elif cmd == 1:
            stats = log_stats.log()
            stats.update(recv)
            conn_send.put(stats)


class LogTopKproc:
    def __init__(self, args: Namespace):
        self.log_stats_remote_send = log_stats_remote_send = Queue()
        self.log_stats_remote_recv = log_stats_remote_recv = Queue()
        log_proc_stats = Process(
            target=log_stats_remote,
            args=(log_stats_remote_send, log_stats_remote_recv, args.main.log_topk,
                  args.main.log_dockscore)
        )
        log_proc_stats.start()
        self.recv_logtop = 0
        self.sent_logtop = 0
        self.log_topk_freq = args.main.log_topk_freq
        self._last_topk_step = 0
        self._no_score_infos = 0
        self._none_score_infos = 0
        self._collected = 0
        self._with_scores = 0
        self._req_stats = {}
        self._all_time_collected = 0

    def collect(self, infos: List[dict], step_log: dict):
        """
            Send infos for good candidates to keep buffer for topK
            Dict should have following keys with values ("proxy", "qed", "synth", "smiles", "score")
            TopK will be chosen based on the score value

            total_num_steps: number of steps at collection time
        """

        # If proxy exist -> rest of the scores exist
        send_info = []
        for sinfo in infos:
            self._collected += 1
            self._all_time_collected += 1
            send_info.append(sinfo)

            # Time to log new info
            if self._all_time_collected >= self._last_topk_step + self.log_topk_freq:
                # Send molecules collected so far
                self.log_stats_remote_send.put((0, (send_info,)))
                send_info = []

                # So we can sync plots at same time
                step_log["approx_step"] = self._last_topk_step + self.log_topk_freq
                step_log["eval_step"] = self._all_time_collected
                step_log["num_sampled_mols"] = self._all_time_collected

                # send request for logs for all new mols so far
                self.send_stats_request(self._all_time_collected, step_log)
                while self._all_time_collected >= self._last_topk_step + self.log_topk_freq:
                    self._last_topk_step += self.log_topk_freq

        if len(send_info) > 0:
            self.log_stats_remote_send.put((0, (send_info,)))

    def get_stats(self):
        """ Non-blocking get """
        log_stats_remote_recv = self.log_stats_remote_recv
        ret_log_stats = []
        while not log_stats_remote_recv.empty():
            log_stats = log_stats_remote_recv.get()
            log_stats.update(self._req_stats.pop(log_stats["eval_step"]))
            ret_log_stats.append(log_stats)
            self.recv_logtop += 1

        return ret_log_stats

    def _local_stats(self):
        collected = self._collected
        stats = {
            "topk_received": collected,
        }
        self._collected = 0
        return stats

    def send_stats_request(self, step: int, extra_log: dict):
        """
            Non blocking send request. LogTopStats will start calculating TopK with current buffer
        """
        if (self.sent_logtop - self.recv_logtop) < 2:
            self._req_stats[step] = self._local_stats()
            self.log_stats_remote_send.put((1, extra_log))
            self.sent_logtop += 1
        else:
            print(f"NOT GOOD. Skipping log top {step}")


def show_histogram(data, scale, edges, zrange=None):
    data = np.stack(data).transpose()
    data = data[::-1, :]
    scaled_data = data / scale
    if zrange is None:
        zrange = [None, None]

    fig = px.imshow(scaled_data, y=edges[::-1].astype(np.str), zmin=zrange[0], zmax=zrange[1])

    fig.add_trace(go.Heatmap(
        # x=list(range(evalhs.shape[0])),
        # y=edges[::-1],
        z=scaled_data,
        customdata=data,
        hovertemplate='Step: %{x}<br>Bin: %{y}<br>Num samples:%{customdata:.3f} <br><b>Prob:%{z:.3f}</b><br>',
        coloraxis="coloraxis1", name=''),
        1, 1)
    return fig
