from typing import List, Any
import numpy as np
import time


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
