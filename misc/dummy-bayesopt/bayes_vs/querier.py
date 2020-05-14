
import copy
from dataclasses import dataclass, field
import heapq


@dataclass(order=True)
class OracleResult:
    score: float  # what we try to minimize!
    smiles: str = field(compare=False)


class QueriedHeap:
    def __init__(self):
        self.queries = []
        self.queries_in_time_order = []

    def add_result(self, result: OracleResult):
        """
        :param result:
        """
        heapq.heappush(self.queries, result)
        self.queries_in_time_order.append(result)

    def pop_off_batch(self, batch_size):
        if len(self.queries) == 0:
            return IndexError
        result = []
        try:
            for i in range(batch_size):
                result.append(heapq.heappop(self.queries))
        except IndexError:
            pass
        return result

    @property
    def all_smiles_set(self):
        return set([el.smiles for el in self.queries])

    @property
    def all_smiles_sorted(self):
        queries = copy.copy(self.queries)
        while True:
            try:
                yield heapq.heappop(queries).smiles
            except IndexError:
                break

    @property
    def smi_to_value_dict(self):
        return {el.smiles: el.score for el in self.queries}

    def __len__(self):
        return len(self.queries_in_time_order)

