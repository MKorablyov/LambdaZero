import numpy as np
import torch
import pandas as pd

from lightrl.a2c_ppo_acktr import utils
from lightrl.a2c_ppo_acktr.envs import make_vec_envs
from lightrl.env.vec_env import get_envs

import LambdaZero
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()


def calculate_topk_match(state_data, logits, act_mask, topk=[1, 5, 10, 20]):
    scores = dict()
    sort_pred = np.argsort(logits)[::-1]

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
        topk_c_f = len(set.intersection(topk_pred_c_g, topk_tgt_cand_g)) / float(
            len(topk_tgt_cand_g))
        scores[f"top{tk}_reward"] = topk_r_f
        scores[f"top{tk}_candidates"] = topk_c_f

    return scores


class EvaluateBase:
    def __init__(self, actor_critic, process_obs, eval_cfg, eval_log_dir, device):

        eval_envs, eval_chunk_size = get_envs(eval_cfg, eval_cfg.procs, send_reset_info=True)

        self.actor_critic = actor_critic
        self.process_obs = process_obs
        self.eval_envs = eval_envs
        self.eval_log_dir = eval_log_dir
        self.device = device
        self.eval_cfg = eval_cfg
        self.eval_batches = getattr(eval_cfg, "eval_batches", 1)

        self._actual_seeds = eval_cfg.env_args.env_seed
        if isinstance(self._actual_seeds, list):
            self.env_seeds = len(self._actual_seeds)
        else:
            self.env_seeds = 1

        self._live_eval = True
        self._log_eval_step = 0

        # Load evaluation data
        if self._live_eval:
            self._eval_data = self._load_eval_data(recalculate_r=False)
            self._eval_topk = [1, 3, 5]
            self._topk_scores = None

    def _load_eval_data(self, recalculate_r=False):
        data = np.load(f"{datasets_dir}/eval_env_data.npy", allow_pickle=True).item()  # type: dict

        # filter eval seeds only
        seeds = self._actual_seeds
        for smi in list(data.keys()):
            if data[smi]["env_seed"][0] not in seeds:
                data.pop(smi)

        # ==========================================================================================
        # Get top r act over dataset
        top_r_p = 0.01
        smi_keys = np.array(list(data.keys()))
        all_r, all_r_act, all_r_k = [], [], []
        for ismi, smi in enumerate(smi_keys):
            rsss = data[smi]["next"]["true_r"].values
            all_r.append(rsss)
            all_r_act.append(data[smi]["next"].index.values)
            all_r_k += [ismi] * len(rsss)
        all_r, all_r_act = np.concatenate(all_r), np.concatenate(all_r_act)
        all_r_k = np.array(all_r_k)
        top_r_ids = np.argsort(all_r)[::-1][:int(top_r_p * len(all_r))]
        top_smi_ids = all_r_k[top_r_ids]
        top_smi_act = all_r_act[top_r_ids]
        for i_smi in np.unique(top_smi_ids):
            data[smi_keys[i_smi]]["dataset_top_r_act"] = top_smi_act[top_smi_ids == i_smi]
        # ==========================================================================================

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

    def eval_step(self, smi, action_dist):
        state_data = self._eval_data[smi]
        topk = self._eval_topk

        act_mask = state_data["action_mask"]
        sort_pred = np.argsort(action_dist)[::-1]

        scores = dict()
        if len(set(state_data["target_r_groups"][:1])) <= 0:
            return dict()

        max_r_group = state_data["next"].true_r_group.max()
        top_act = state_data["next"][state_data["next"].true_r_group == max_r_group].index.values
        act_order = sort_pred.tolist()
        num_act = len(act_order)
        top_pos = []
        top_pos_logit = []
        for b_act in top_act:
            pos = act_order.index(b_act)
            top_pos.append(max(0, pos - len(top_act) + 1))

        probs = torch.softmax(torch.from_numpy(action_dist), 0)
        scores[f"argmax_pos"] = np.mean(top_pos)
        scores[f"argmax_cnt"] = len(top_act)
        scores[f"logit_max"] = action_dist.max()
        scores[f"argmax_logit_dist"] = action_dist.max() - np.mean([action_dist[x] for x in top_act])
        scores[f"argmax_prob"] = np.mean([probs[x].item() for x in top_act])

        # Calculate scores for top r across dataset
        if "dataset_top_r_act" in state_data:
            dataset_top_r_act = state_data["dataset_top_r_act"]
            top_dataset_pos = np.array([act_order.index(x) for x in dataset_top_r_act])
            top_dataset_pos = np.clip(top_dataset_pos - len(dataset_top_r_act) + 1, 0, np.inf)
            scores[f"argmax_D_pos"] = np.mean(top_dataset_pos)
            scores[f"argmax_D_prob"] = np.sum([probs[x].item() for x in dataset_top_r_act])

        # ADD to scores
        topk_info = calculate_topk_match(state_data, action_dist, act_mask, topk=topk)
        scores.update(topk_info)
        return scores

    def evaluate(self, train_iter: int, train_steps: int):
        eval_envs = self.eval_envs
        env_seeds = self.env_seeds
        eval_batches = self.eval_batches
        process_obs = self.process_obs
        actor_critic = self.actor_critic
        device = self.device
        eval_episode_rewards = [list() for _ in range(env_seeds)]
        ev_scores = [list() for _ in range(env_seeds)]
        arg_max_act = [list() for _ in range(env_seeds)]
        num_eval_ep = env_seeds * (eval_batches + 1)
        num_processes = eval_envs.num_procs

        obs = eval_envs.reset(num_evals=num_eval_ep)

        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)

        while np.any([len(x) < eval_batches for x in eval_episode_rewards]):
            p_obs = process_obs(obs)

            with torch.no_grad():
                res_m = actor_critic.act(
                    p_obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=True)
                action, eval_recurrent_hidden_states = res_m.action, res_m.rnn_hxs

            # Obser reward and next obs
            send_act = action.flatten().cpu().numpy().tolist()

            obs, reward, done, infos = eval_envs.step(send_act)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            act_log_probs = res_m.dist.logits.data.cpu().numpy()

            for iobs, info in enumerate(infos):
                if done[iobs]:
                    ep_seed = info["episode"] % env_seeds
                    eval_episode_rewards[ep_seed].append(reward[iobs])

                if "act_molecule" in info:
                    score = self.eval_step(info["act_molecule"], act_log_probs[iobs])
                    ev_scores[ep_seed].append(score)
                    arg_max_act[ep_seed].append(act_log_probs[iobs].argmax())

        flat_s = [item for sublist in ev_scores for item in sublist[:eval_batches]]

        all_scores = pd.DataFrame.from_dict(flat_s)

        # ==========================================================================================
        # Calc argmax act stats
        flat_arg_max_act = np.array([item for sublist in arg_max_act for item in sublist[:eval_batches]])

        max_blocks = 7
        num_blocks = 105

        add_act = flat_arg_max_act[flat_arg_max_act >= max_blocks]
        stem_idx = (add_act - max_blocks) // num_blocks
        block_idx = (add_act - max_blocks) % num_blocks
        all_scores["unique_stem_idx"] = len(np.unique(stem_idx))
        all_scores["unique_block_idx"] = len(np.unique(block_idx))
        for act in range(max_blocks):
            all_scores[f"cnt_act_{act}"] = (flat_arg_max_act == act).sum()

        # ==========================================================================================

        print(all_scores.mean())

        flat_r = [item for sublist in eval_episode_rewards for item in sublist[:eval_batches]]
        print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
            len(flat_r), np.mean(flat_r)))

        return all_scores
