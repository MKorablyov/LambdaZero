import copy
import os
import time
import numpy as np
import ray
from rdkit import Chem
from rdkit.Chem import QED
from datetime import datetime
import shortuuid
import glob
from typing import List, Tuple
import re
import pandas as pd
import gc
import json

from LambdaZero.utils.actor_pool_wait import ActorPoolWait
from LambdaZero.chem import DockVina_smi
from LambdaZero.models import ChempropWrapper_v1
from LambdaZero.utils import get_external_dirs
import os.path as osp
import LambdaZero

datasets_dir, programs_dir, summaries_dir = get_external_dirs()


config_DockingOracle_v1 = {"num_threads":8,
                           "dockVina_config": {"outpath": osp.join(summaries_dir, "docking")},
                           "mean":-8.6, "std": 1.1, "act_y":LambdaZero.contrib.functional.elu2,
                           }


class PreDockingDB:
    """
        Class used to load HDF5 db with molecule data and keep a dict(smiles, dockscore) in memory.
        Local dict is updated during runtime with new oracle dock scores (not the database!)
        call class to get pre-docked scores
        Database, contains other data:  [smiles, dockscore, blockidxs, slices, jbonds, stems]
            store = pd.HDFStore(best_db)
            db = store.select('df') # DataFrame with all data

        Tested: 10 mil (fake) mol DB
            -> 1.1G on disk
            -> 5.G in mem
            -> 2.2G in mem only using dict (after __init__)
            -> time to load 31.13s (and get dict)
            -> query speed 2.71e-08s
    """

    def __init__(self):
        self._db_hits = 0
        best_db, _ = PreDockingDB.get_last_db()
        if best_db is not None:
            # Read database
            store = pd.HDFStore(best_db, "r")
            db = store.select('df')
            self.pre_dock_scores = db["dockscore"].to_dict()

            # Free up mem
            store.close(); del db; del store; gc.collect()
            print(f"Loaded {best_db} with  {len(self.pre_dock_scores)} pre-docked scores")
        else:
            self.pre_dock_scores = dict()
            print(f"No previous DB for pre-docked scores")

    def __call__(self, smiles: str) -> float:
        """ Get dock score if exists in dict """
        res = self.pre_dock_scores.get(smiles)
        predocked_score = res if res is None else np.around(np.float64(res), 1)

        return predocked_score

    def local_update(self, smiles: str, dockscore: float):
        """
            Used for live update of in memory db!
            Does NOT actually update the db on disk
        """
        self.pre_dock_scores[smiles] = dockscore

    @staticmethod
    def get_dock_db_base_path() -> str:
        dock_db_dir = os.path.join(datasets_dir, "dock_dbs")
        if not os.path.isdir(dock_db_dir):
            os.mkdir(dock_db_dir)

        return os.path.join(dock_db_dir, "dock_db")

    @staticmethod
    def get_db_scores_base_path() -> str:
        db_dir = os.path.join(datasets_dir, "dock_db_scores")
        if not os.path.isdir(db_dir):
            os.mkdir(db_dir)

        return db_dir

    @staticmethod
    def get_db_scores_path(db_path: str) -> str:
        name = os.path.basename(db_path).split(".")[0]
        return f"{PreDockingDB.get_db_scores_base_path()}/{name}.pk"

    @staticmethod
    def get_new_db_name() -> str:
        tp = time.time()
        date_time = datetime.fromtimestamp(tp)
        d = date_time.strftime("%Y_%m_%d_%Hh")
        path = f"{PreDockingDB.get_dock_db_base_path()}_{int(tp)}tp_{d}.h5"
        return path

    @staticmethod
    def get_last_db() -> Tuple[str, List[str]]:
        dbs = glob.glob(f"{PreDockingDB.get_dock_db_base_path()}_*.h5")
        dbs = sorted(dbs, key=lambda x: int(re.findall("dock_db_(.*)tp_", x)[0]))
        if len(dbs) == 0:
            return None, dbs

        return dbs[-1], dbs

    @staticmethod
    def read_last_db(sample_size: int = 0, with_scores: bool = False) -> pd.DataFrame:
        pathdb, all_dbs = PreDockingDB.get_last_db()
        columns = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        store = pd.HDFStore(pathdb, "r")
        df = store.select('df')  # type: pd.DataFrame

        if sample_size != 0:
            df = df.iloc[:sample_size]

        for cl_mame in columns[2:]:
            df.loc[:, cl_mame] = df[cl_mame].apply(json.loads)
        df.dockscore = df.dockscore.apply(lambda x: np.around(np.float64(x), 1))

        if with_scores:
            score_path = PreDockingDB.get_db_scores_path(pathdb)
            assert os.path.isfile(score_path), f"Did not found scores for {pathdb}"
            db_scores = pd.read_pickle(score_path)

            if sample_size != 0:
                db_scores = db_scores.iloc[:sample_size]

            assert (df.index.values == db_scores.smiles.values).all(), "Not same smiles"

            df["qed_score"] = db_scores["qed_score"].values
            df["synth_score"] = db_scores["synth_score"].values

        return df


class DockVina_smi_db(DockVina_smi):
    """
        Dump Docking data to file.
        data ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        This files will be merged in the database
    """
    def __init__(self, *args, save_data: bool = True, query_predocked: bool = True, **kwargs):
        DockVina_smi.__init__(self, *args, **kwargs)

        self.save_data = save_data
        fld_out = os.path.join(summaries_dir, "docking_dump")

        if not os.path.isdir(fld_out):
            os.mkdir(fld_out)

        data_out = f"{fld_out}/data_{int(time.time())}_{shortuuid.uuid()}"

        # Seems rudimentary, writing to text file, but it seems the fastest way in order to flush
        # (thus be robust to crashes).
        # Estimated avg ~3s / 10k rows to write
        self.file_header = ["smiles", "dockscore", "blockidxs", "slices", "jbonds", "stems"]
        self.file_object = open(f"{data_out}.log", "a")
        self.file_object.write(str(self.file_header) + "\n")
        self.file_object.flush()

        self.query_predocked = query_predocked
        if query_predocked:
            self.predocked = PreDockingDB()

    def dock(self, smiles, mol_data: dict = None, **kwargs):
        if self.query_predocked:
            res = self.predocked(smiles)
            if res is not None:
                return None, res, None

        # It is run if no pre-docking is used or no pre-docked smile was found
        mol_name, dockscore, coord = super().dock(smiles, **kwargs)

        if self.query_predocked:
            self.predocked.local_update(smiles, dockscore)

        if self.save_data and mol_data is not None:
            data = [smiles, dockscore]
            data += [mol_data[k] for k in ["blockidxs", "slices", "jbonds", "stems"]]
            self.file_object.write(str(data)+"\n")
            self.file_object.flush()
        return mol_name, dockscore, coord


@ray.remote(num_gpus=0)
class DockingEstimator(DockVina_smi_db):
    def __init__(self, dockVina_config):
        DockVina_smi_db.__init__(self, **dockVina_config)

    def eval(self, smiles, **kwargs):
        try:
            mol_name, dockscore, coord = self.dock(smiles, **kwargs)
        except Exception as e:
            dockscore = None

        return dockscore


class SimpleDockBuffer:
    def __init__(self, buffer_size=1000):
        self._max_buffer = buffer_size
        self._buffer = []
        self._cnt = 0

    def __call__(self, values):
        self._buffer += [_x for _x in values if _x is not None]
        self._cnt += len(values)

    def is_full(self):
        return self._cnt > self._max_buffer

    def empty_buffer(self):
        ret = self._buffer, self._cnt
        self._buffer, self._cnt = [], 0
        return ret


class DockingOracle:
    def __init__(self, num_threads, dockVina_config, mean, std, act_y, logger, max_failed_hits=3):
        self.num_threads = num_threads

        # Create docked data storage
        self.query_predocked = query_predocked = getattr(dockVina_config, "query_predocked", True)
        dockVina_config["query_predocked"] = False  # Make sure actors don't load db as well
        if query_predocked:
            print("Using pre-docked query")
            self.predocked = PreDockingDB()

        # create actor pool
        self.actors, self.pool = [], None
        if self.num_threads > 0:
            self.actors = [DockingEstimator.remote(dockVina_config) for i in range(self.num_threads)]
            self.pool = ActorPoolWait(self.actors, timeout=360)

        self.mean = mean
        self.std = std
        self.act_y = act_y
        self.fix_none = True
        self.norm_data = True
        self.logger = logger
        self._simple_buffer = SimpleDockBuffer()
        self._max_failed_hits = max_failed_hits
        self._failed_hits = dict()

    def set_fix_none(self, vl):
        self.fix_none = vl

    def set_norm_data(self, vl):
        self.norm_data = False

    def __call__(self, data, ret_true_dockscore=False):
        filter_failed = [False] * len(data)

        if self.query_predocked:
            predockedscores, smiles_ids, smiles = [None] * len(data), [], []
            fh = self._failed_hits
            fh_max = self._max_failed_hits

            # Get predocked scores if not add them for oracle
            for i, d in enumerate(data):
                res = self.predocked(d["smiles"])
                if res is None:
                    if d["smiles"] in fh and fh[d["smiles"]] > fh_max or fh_max == 0:
                        # Filter Failed hits... sadly add average
                        predockedscores[i] = self.mean
                        filter_failed[i] = True
                    else:
                        smiles.append((d["smiles"], d))
                        smiles_ids.append(i)
                else:
                    predockedscores[i] = res

            # Get oracle scores
            if len(smiles) > 0 and len(self.actors) > 0:

                dockscores = list(self.pool.map(
                    lambda actor, smi: actor.eval.remote(smi[0], mol_data=smi[1]), smiles
                ))

                # Log them in place
                for i, ds in zip(smiles_ids, dockscores):
                    if ds is not None:
                        predockedscores[i] = ds
                        self.predocked.local_update(data[i]["smiles"], ds)
                    else:
                        if data[i]["smiles"] in fh:
                            fh[data[i]["smiles"]] += 1
                        else:
                            fh[data[i]["smiles"]] = 1

            dockscores = predockedscores
        else:
            smiles = [(d["smiles"], d) for d in data]
            dockscores = list(self.pool.map(
                lambda actor, smi: actor.eval.remote(smi[0], mol_data=smi[1]), smiles
            ))

        true_dockscores = [None if filter_failed[ix] else x for ix, x in enumerate(dockscores)]
        dockscores_ = []
        num_failures = 0
        for d in dockscores:
            if d == None and self.fix_none:
                dockscores_.append(self.mean) # mean on failures
                num_failures+=1
            else:
                dockscores_.append(d)

        if self.norm_data and self.fix_none:
            dockscores = [(self.mean-d) / self.std for d in dockscores_] # this normalizes and flips dockscore
            dockscores = self.act_y(dockscores)

        self._simple_buffer([x for ix, x in enumerate(dockscores) if not filter_failed[ix]])

        if self._simple_buffer.is_full():
            log_dock, log_cnt = self._simple_buffer.empty_buffer()
            self.logger.log.remote({
                "docking_oracle/failure_probability": (log_cnt-len(log_dock)) / float(log_cnt),
                "docking_oracle/norm_dockscore_min": np.min(log_dock),
                "docking_oracle/norm_dockscore_mean": np.mean(log_dock),
                "docking_oracle/norm_dockscore_max": np.max(log_dock)})

        if ret_true_dockscore:
            return dockscores, true_dockscores

        return dockscores

    def update_predocked(self, smiles: str, dockscore: float):
        if dockscore is None:
            # Update failed hits
            if smiles in self._failed_hits:
                self._failed_hits[smiles] += 1
            else:
                self._failed_hits[smiles] = 1
        else:
            self.predocked.local_update(smiles, dockscore)

@ray.remote
class QEDEstimator:
    def __init__(self):
        pass

    def eval(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            qed = QED.qed(mol)
        except Exception as e:
            qed = 0.0
        return qed

class QEDOracle:
    def __init__(self, num_threads):
        self.num_threads = num_threads
        # create actor pool
        self.actors = [QEDEstimator.remote() for i in range(self.num_threads)]
        self.pool = ray.util.ActorPool(self.actors)

    def __call__(self, molecules):
        smiles = [m["smiles"] for m in molecules]
        qeds = list(self.pool.map(lambda actor, smi: actor.eval.remote(smi), smiles))
        return qeds


@ray.remote(num_gpus=0.05)
class ChempropWrapper_v2(ChempropWrapper_v1):
    def eval(self, m):
        return ChempropWrapper_v1.__call__(self, m)

class SynthOracle:
    def __init__(self, synth_options, synth_config):
        self.synth_net = ChempropWrapper_v2.options(**synth_options).remote(synth_config)

    def __call__(self,molecules):
        synths = []
        for m in molecules:
            try:
                synth = ray.get(self.synth_net.eval.remote(m["mol"]))
            except Exception as e:
                print(e)
                synth = 0.0
            synths.append(synth)
        return synths