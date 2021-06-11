import os
from copy import deepcopy
import numpy as np
import torch

from lightrl.env.block_mol_graph_v1 import BlockMolEnvGraph_v1
import LambdaZero

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


class BlockMolEnvGraph_Eval(BlockMolEnvGraph_v1):
    def __init__(self, config=None, *args, **kwargs):
        self.worker_index = getattr(config, "worker_index", 0)
        self.vector_index = getattr(config, "vector_index", 0)
        config["worker_index"] = self.worker_index
        config["vector_index"] = self.vector_index
        self._step_cnt = 0
        self._reset_mode = False

        super(BlockMolEnvGraph_Eval, self).__init__(config=config, *args, **kwargs)

        # Save config info
        config = self._config
        self._logger = getattr(config, "logger", None)
        self._eval_mode = config["eval_mode"]
        self._save_eval_data = config["save_eval_data"]
        self._filter_init_smi = []  # Init env smiles to filter out
        self._log_dir = getattr(config, "log_dir", f"{summaries_dir}/results")

        # Config env seeds based on worker index
        self.set_seed(config, worker_index=self.worker_index, vector_index=self.vector_index)

        # Filter out init seeds for training
        if not self._eval_mode and len(config["filter_init_states"]) > 0:
            self._calculate_ignored_init_smiles(config["filter_init_states"])

        if self._eval_mode:
            assert config["evaluation_num_episodes"] is not None, \
                "Must set evaluation_num_episodes! in order to log each evaluation cycle correctly"

            self._log_eval_freq = config["evaluation_num_episodes"]
            self._live_eval = True

            self._log_eval_step = 0
            self._prev_training_iteration = 0
            self._prev_timesteps_total = 0

            self._log_data = dict({
                "step": [], "pre_batch": [], "post_batch": [], "batch_step": [], "other_info": []
            })

            if self._save_eval_data:
                self._save_batch = 0

                # Create eval folder path
                self._log_dir_eval = f"{self._log_dir}/eval_data"
                if not os.path.isdir(self._log_dir_eval):
                    os.mkdir(self._log_dir_eval)

            # Load evaluation data
            if self._live_eval:
                self._eval_data = self._load_eval_data(recalculate_r=False)
                self._eval_topk = [1, 3, 5]
                self._topk_scores = None

    def _calculate_ignored_init_smiles(self, reject_seeds):
        filter_smiles = self._filter_init_smi
        config = self._config

        backup_env_seed = deepcopy(self._env_seed)
        backup_config = deepcopy(config["env_seed"])

        # Calculate starting smiles to reject based on init seed
        for seed in reject_seeds:
            config["env_seed"] = seed
            self.set_seed(config, worker_index=self.worker_index, vector_index=self.vector_index)
            obs = super().reset()
            filter_smiles.append(self.molMDP.molecule.smiles)

        # Reset to backup
        self._env_seed = backup_env_seed
        config["env_seed"] = backup_config
        self.set_seed(config, worker_index=self.worker_index, vector_index=self.vector_index)

        print("Filter smiles:", filter_smiles)

    def _calc_true_r(self, dock_score, synth, qed):
        from LambdaZero.contrib.functional import satlins
        r_config = self._config["reward_config"]

        qed_cutoff = r_config["qed_cutoff"]
        synth_cutoff = r_config["synth_cutoff"]
        mean = r_config["scoreProxy_config"]["oracle_config"]["mean"]
        std = r_config["scoreProxy_config"]["oracle_config"]["std"]
        act_y = r_config["scoreProxy_config"]["oracle_config"]["act_y"]

        clip_qed = satlins(qed, qed_cutoff[0], qed_cutoff[1])
        clip_synth = satlins(synth, synth_cutoff[0], synth_cutoff[1])
        proxy_dock = mean if np.isnan(dock_score) else dock_score
        proxy_dock = (mean-proxy_dock) / std
        proxy_dock = act_y([proxy_dock])[0]
        reward = float(proxy_dock) * clip_qed * clip_synth

        return reward

    def _load_eval_data(self, recalculate_r=False):
        data = np.load(f"{datasets_dir}/eval_env_data.npy", allow_pickle=True).item()  # type: dict

        # filter eval seeds only
        seeds = self._config["env_seed"]
        for smi in list(data.keys()):
            if data[smi]["env_seed"][0] not in seeds:
                data.pop(smi)

        if recalculate_r:
            r_group_dec = 3

            # Recalculate R from pre-saved dock, synt and qed scores
            for smi, smi_data in data.items():
                df = smi_data["next"]

                # Recalculate Reward
                df["true_r"] = df.apply(lambda row: self._calc_true_r(
                    row["true_dockscore"], row["synth_score"], row["qed_score"]
                ), axis=1)

                # Calculate order of target R groups
                df["true_r_group"] = df.groupby(np.around(df["true_r"], r_group_dec)).ngroup()
                smi_data["target_r_groups"] = \
                    df.sort_values(by="true_r")["true_r_group"].values[::-1]

        return data

    def eval_step(self, smi, action_dist):
        state_data = self._eval_data[smi]
        topk = self._eval_topk

        act_mask = state_data["action_mask"]
        sort_pred = np.argsort(action_dist)[::-1]

        scores = dict()
        if len(set(state_data["target_r_groups"][:1])) <= 0:
            return dict()

        for tk in topk:
            topk_sort_pred = sort_pred[:tk]
            sort_available_pred = topk_sort_pred[act_mask[topk_sort_pred]]
            topk_pred_data = state_data["next"].loc[sort_available_pred]
            topk_pred_r_g = set(topk_pred_data["true_r_group"].values)
            topk_pred_c_g = set(topk_pred_data["candidate_dockscore_group"].values)

            topk_tgt_r_g = set(state_data["target_r_groups"][:tk])
            topk_tgt_cand_g = set(state_data["target_candidate_groups"][:tk])

            # if topk == 3:
            #     print(topk_tgt_r_g)
            topk_r_f = len(set.intersection(topk_pred_r_g, topk_tgt_r_g)) / float(len(topk_tgt_r_g))
            topk_c_f = len(set.intersection(topk_pred_c_g, topk_tgt_cand_g)) / float(len(topk_tgt_cand_g))
            scores[f"top{tk}_reward"] = topk_r_f
            scores[f"top{tk}_candidates"] = topk_c_f

        return scores

    @staticmethod
    def get_mol_data(molecule) -> dict:
        return deepcopy({
            "blockidxs": molecule.blockidxs,
            "slices": molecule.slices,
            "jbonds": molecule.jbonds,
            "stems": molecule.stems,
            "smiles": molecule.smiles,
        })

    def step(self, action):
        if self._eval_mode and not self._reset_mode:
            # Pre step data for identification
            data = dict({"obs_cnt": self._step_cnt, "action": action,
                         **self.get_mol_data(self.molMDP.molecule)})

        # Step
        obs, reward, done, info = super().step(action)

        if self._eval_mode and not self._reset_mode:
            # Post step data
            data.update(dict({"n_obs": obs, "n_reward": reward, "n_done": done, "n_info": info}))
            self._log_data["step"].append(data)

        self._step_cnt += 1

        return obs, reward, done, info

    def add_eval_data(self, pre_batch, post_batch):
        if not self._eval_mode:
            return
        s_data = self._log_data

        if self._live_eval:
            # Last save data should be from this step - We could double check this though
            assert len(pre_batch["action_dist_inputs"]) == 1, "Parallel eval envs not implemented"
            rsc = self.eval_step(s_data["step"][-1]["smiles"], pre_batch["action_dist_inputs"][0])

            if self._topk_scores is None and len(rsc) > 0:
                self._topk_scores = dict({x: [] for x in rsc.keys()})

            for k, v in rsc.items():
                self._topk_scores[k].append(v)

        s_data["pre_batch"].append(pre_batch)
        s_data["post_batch"].append(post_batch)
        s_data["batch_step"].append(self._step_cnt)
        s_data["other_info"].append({
            "training_iteration": self._prev_training_iteration,
            "timesteps_total": self._prev_timesteps_total,
        })
        training_iteration = self._prev_training_iteration
        timesteps_total = self._prev_timesteps_total

        self._log_eval_step += 1

        if self._log_eval_step % self._log_eval_freq == 0:
            # Every end of evalution cycle

            if self._live_eval:
                eval_log_data = dict({
                    "training_iteration": training_iteration,
                    "timesteps_total": timesteps_total
                })
                for k, v in self._topk_scores.items():
                    print(f"EVAL Results: {k}: {np.mean(v)}")
                    eval_log_data[f"evaluation/{k}"] = np.mean(v)
                    v.clear()

                self._logger.log.remote(eval_log_data)

            # All env seeds have been tested
            self._reset_eval_data()

    def _reset_eval_data(self):
        """ Reset buffer of evaluated data """

        if self._save_eval_data:
            st_it = self._log_data["other_info"][0]["training_iteration"]

            step_path = f"{self._log_dir_eval}/batch_{self._save_batch}_train_iter_{st_it}"
            torch.save(self._log_data, step_path)
            self._save_batch += 1

        for k, v in self._log_data.items():
            v.clear()

    def reset(self):
        if len(self._filter_init_smi) > 0:
            # Try for a couple of times to reset to different state
            for _ in range(5):
                obs = super().reset()
                if self.molMDP.molecule.smiles not in self._filter_init_smi:
                    return obs
            assert False, f"Too many tries to find a init state out of {self._filter_init_smi}, " \
                          f"(change seed of either eval or train envs)"
        obs = super().reset()
        return obs


