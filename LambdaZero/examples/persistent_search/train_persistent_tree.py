
import warnings
warnings.filterwarnings('ignore')
import sys
import time
import os
import os.path as osp
import pickle
import gzip
import psutil

import ray
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, Set2Set


from LambdaZero.chem import atomic_numbers
from LambdaZero.environments.persistent_search import PersistentSearchTree, PredDockRewardActor, SimDockRewardActor, RLActor, MBPrep, RandomRLActor
import LambdaZero.utils
datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()

warnings.filterwarnings('ignore')


class MPNNet_v2(nn.Module):
    def __init__(self, num_feat=14, dim=64, num_out_per_stem=105, num_stems=8):
        super().__init__()
        self.lin0 = nn.Linear(num_feat, dim)
        self.num_ops = num_out_per_stem
        self.num_stems = num_stems

        net = nn.Sequential(nn.Linear(4, 128), nn.LeakyReLU(), nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr='mean')
        self.gru = nn.GRU(dim, dim)

        self.lin1 = nn.Linear(dim, dim * 8)
        self.lin2 = nn.Linear(dim * 8, num_out_per_stem)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, 1)

    def forward(self, data):
        out = F.leaky_relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(12):
            m = F.leaky_relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            out = out.squeeze(0)

        data.x = self.lin2(F.leaky_relu(self.lin1(out)))
        add_logits = torch.zeros((data.num_graphs, self.num_stems, self.num_ops),
                                 device=out.device)
        for i, g in enumerate(data.to_data_list()):
            add_logits[i, :len(g.stem_idx)] = g.x[g.stem_idx]

        out = self.set2set(out, data.batch)
        r = self.lin3(out)[:, 0]
        return r, add_logits.reshape((add_logits.shape[0], -1))


class MolAC_GCN(nn.Module):
    def __init__(self, obs_config):
        nn.Module.__init__(self)
        act = nn.LeakyReLU()

        self.mpnn = MPNNet_v2(
            num_feat=14 + 1 + len(atomic_numbers),
            dim=64,
            num_out_per_stem=obs_config['num_blocks'],
            num_stems=obs_config['max_branches'])


    def forward(self, graph, action_mask):
        r, add_logits = self.mpnn(graph)
        masked_actions = (1. - action_mask).to(torch.bool)
        add_logits[masked_actions] = -20 # some very small prob that does not lead to inf

        return r, add_logits

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


def guided_search(exp_hps):
    device = torch.device('cuda')


    salt = hex(abs(hash(str(exp_hps))))[2:6]
    exp_path = time.strftime(f'{exp_hps["save_path"]}/guided_search_max_%m%b_%d_%H_%M_{salt}')

    hyperparameters = {
        'actor_temperature': 1, # temperature of the boltzmann Q policy
        'score_temperature': 1.5, # priority temperature for sampling nodes
        'priority_pred': 'greedy_q', # greedy or boltzmann
        'return_type': 'montecarlo', # MonteCarlo vs Max descendant reward
        'num_molecules': int(10e3), # Total number of molecules to try
        'prune_at': int(5e3), # Prune the search tree when it is this big
        'update_prio_on_refresh': False,
        'num_rollout_actors': 0,
        'num_docking_threads': 1,
    }
    hyperparameters.update(exp_hps)



    obs_config = {"mol_fp_len": 512,
                  "mol_fp_radiis": [3],
                  "stem_fp_len": 64,
                  "stem_fp_radiis": [4, 3, 2],
    }

    from LambdaZero.examples.synthesizability.vanilla_chemprop import DEFAULT_CONFIG as chemprop_cfg

    reward_config = {
        "soft_stop": True,
        "load_model": osp.join(datasets_dir, "brutal_dock/d4/dock_blocks105_walk40_12_clust_model002"),
        "natm_cutoff": [45, 50],
        "qed_cutoff": [0.2, 0.7],
        "synth_cutoff":[0, 5],
        "exp": None,
        "delta": False,
        "simulation_cost": 0.00,
        "device": device,
        "synth_config": chemprop_cfg,
    }

    score_temperature = hyperparameters['score_temperature']
    if hyperparameters['priority_pred'] in ['greedy_q', 'boltzmann']:
        score_fn = lambda n: np.exp(n.value / score_temperature)
    elif hyperparameters['priority_pred'] == 'max_desc_r':
        score_fn = lambda n: np.exp(n.max_descendant_r / score_temperature)

    env_config = {
        "blocks_file": osp.join(datasets_dir, "fragdb/blocks_PDB_105.json"),
        "obs_config": obs_config,
        "reward_config": reward_config,
        'exp_path': exp_path,
        'max_steps': 1,
        "num_blocks": 105,
        "max_blocks": 10,
        "max_atoms": 50,
        "max_branches": 20,
        'num_molecules': hyperparameters['num_molecules'],
        #'seeding_nodes': 'top_mols_50k_05_May_02_00_10_22.pkl.gz',
        #'seeding_nodes': 'top_mols_5k_04_Apr_22.pkl.gz',
        #'seeding_nodes': 'top_mols_50k_04_Apr_27.pkl.gz',
        'graph_add_stem_mask': True,
        'return_type': hyperparameters['return_type'],
        'prune_at': hyperparameters['prune_at'],
        #'num_molecules': int(10e6),
        #'prune_at': int(1e3),
        #'score_fn': lambda r, v: np.exp((r*0+v)/score_temperature),
        #'score_fn': lambda n: np.exp((n.value - n.max_descendant_r)/score_temperature),
        'score_fn': score_fn,
        'update_prio_on_refresh': hyperparameters['update_prio_on_refresh'],
    }
    env_config['num_actions'] = env_config['num_blocks'] * env_config['max_branches']


    # Todo: incorporate this into persistent_search
    class DataBatchers:
        def __init__(self, batchers):
            self.batchers = batchers
            self.batches = [i.get.remote() for i in batchers]

        def get(self):
            ready, not_ready = ray.wait(self.batches, 1)
            bidx = self.batches.index(ready[0])
            self.batches[bidx] = self.batchers[bidx].get.remote()
            return ray.get(ready[0])

    obs_config['num_blocks'] = env_config['num_blocks']
    obs_config['max_branches'] = env_config['max_branches']
    mbsize = 64

    pred_dock_actor = PredDockRewardActor.options(max_concurrency=2).remote(**reward_config)
    sim_dock_actor = SimDockRewardActor.options(max_concurrency=2).remote(
        os.environ['SLURM_TMPDIR'],
        programs_dir, datasets_dir,
        hyperparameters['num_docking_threads'],
    )

    tree = ray.remote(PersistentSearchTree).options(max_concurrency=mbsize * 2).remote(env_config)

    time.sleep(0.5)
    avg_q_loss = avg_r_loss = 0
    num_actions = env_config['num_actions']
    ray.get(tree.update_values.remote())

    train_batchers = DataBatchers([MBPrep.remote(env_config, tree, mbsize * 2)
                                  for i in range(2)])
    act_batcher = MBPrep.remote(env_config, tree, mbsize)

    rl_actor = RLActor.options(max_concurrency=2).remote(
        MolAC_GCN, obs_config,
        device, train_batchers, act_batcher, tree,
        hyperparameters['actor_temperature'],
        hyperparameters['priority_pred'])

    rollout_actors = [rl_actor]
    batch_actors = []
    for i in range(hyperparameters['num_rollout_actors']):
        act_batcher_i = MBPrep.remote(env_config, tree, mbsize)
        rl_actor_i = RLActor.options(max_concurrency=2).remote(
            MolAC_GCN, obs_config,
            device, [], act_batcher_i, tree,
            hyperparameters['actor_temperature'],
            hyperparameters['priority_pred'])
        batch_actors.append(act_batcher_i)
        rollout_actors.append(rl_actor_i)


    time.sleep(0.5)
    print("Starting reward threads")
    pred_dock_actor_thread = pred_dock_actor.run.remote(tree, 256)
    sim_dock_actor_thread = sim_dock_actor.run.remote(tree, 12)

    idxs_available_actions = [list(zip(*ray.get(tree.sample_many.remote(mbsize, idxs_and_aa=True))))
                              for i in range(len(rollout_actors))]

    logs = {
        'q_loss': [],
        'r_loss': [],
        'top_10': [],
        'hps': hyperparameters,
    }
    for step in range(env_config['num_molecules'] // mbsize):
        t0 = time.time()
        train_cb = rl_actor.train.remote()
        all_policies = [
            rla.get_pol.remote(idxs, update_r=True)
            for rla, (idxs, available_actions) in zip(rollout_actors, idxs_available_actions)]
        all_policies = ray.get(all_policies)
        # update the newly created nodes that weren't touched by the above update_r
        qidxs = ray.get(tree.pop_from_new_queue.remote())
        if len(qidxs):
            rl_actor.get_pol.remote(qidxs, update_r=True)
        # take actions

        new_iaa = []
        for policies, (idxs, available_actions) in zip(all_policies, idxs_available_actions):
            actions = []
            for aa, pol in zip(available_actions, policies):
                p = np.zeros(pol.shape, dtype=np.float64) # Necessary because of how np.random works
                p[aa] = pol[aa]
                p /= p.sum()
                actions.append(np.random.multinomial(1, p).argmax())
            new_iaa.append(tree.take_actions.remote(idxs, actions, do_restart=True))
        idxs_available_actions = ray.get(new_iaa)
        q_loss, r_loss = ray.get(train_cb)
        avg_q_loss = 0.99 * avg_q_loss + 0.01 * q_loss
        avg_r_loss = 0.99 * avg_r_loss + 0.01 * r_loss
        logs['q_loss'].append(q_loss)
        logs['r_loss'].append(r_loss)
        t1 = time.time()
        if not step % 50:
            top_r = ray.get(tree.get_top_k_nodes.remote(10))
            print(step, env_config['num_molecules'] // mbsize,
                  f'{t1-t0:.3f}',f'{(t1-t0)*1000/(mbsize * len(rollout_actors)):.1f}ms/mol',
                  f'{q_loss:.3f} {r_loss:.3f}',
                  f'{avg_q_loss:.3f} {avg_r_loss:.3f}')
            print(' '.join(f'{i[0]:.3f}' for i in top_r))
            logs['top_10'].append([i[0] for i in top_r])

        if not step % 50:
            nstored = ray.get(tree.get_num_stored.remote())
            if nstored > env_config['prune_at']:
                ray.get(tree.prune_tree.remote())

            for rla in rollout_actors[1:]:
                rla.set_parameters_from.remote(rollout_actors[0])

        if not step % 50:
            tree.update_values.remote()
            if ray.get(tree.is_full.remote()):
                break

            tot_mem = sum([p.memory_info().rss
                           for p in psutil.process_iter(['name', 'username', 'memory_info'])
                           if p.username() == psutil.Process(os.getpid()).username()]) / 1024 / 1024
            print(f'{tot_mem:.1f}M, {tot_mem/1024:.3f}G')

        if step and not step % 2000:
            ray.get(tree.save.remote(exp_path))
            pickle.dump(logs, gzip.open(f'{exp_path}/logs.pkl.gz', 'wb'))
    print('Saving final tree and stopping actors...')
    save = tree.save.remote(exp_path)
    pickle.dump(logs, gzip.open(f'{exp_path}/logs.pkl.gz', 'wb'))
    ray.get(pred_dock_actor.stop.remote())
    ray.get(sim_dock_actor.stop.remote())
    ray.get(save)

if __name__ == '__main__':
    ray.init(num_cpus=12)

    if len(sys.argv) == 2 and sys.argv[1] == 'test':
        for priority_pred in ['greedy_q', 'boltzmann']:
            for return_type in ['montecarlo', 'max_desc_r']:
                hps = {'priority_pred': priority_pred,
                       'return_type': return_type,
                       'save_path': os.environ["SCRATCH"]+'/lz'}
                guided_search(hps)

    elif 'array' in sys.argv:
        configs = [
            {'priority_pred': priority_pred,
             'return_type': return_type,
             'num_molecules': int(2e6),
             'prune_at': prune_at,
             'actor_temperature': actor_temperature,
             'score_temperature': score_temperature,
             'save_path': '/home/bengioem/scratch/lz'}

            for priority_pred in ['greedy_q', 'boltzmann']
            for return_type in ['montecarlo', 'max_desc_r']
            for prune_at in [int(150e3), int(250e3)]
            for actor_temperature in [1, 5]
            for score_temperature in [1.5, 5]
            for run in [0, 1]
        ]
        array_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        guided_search(configs[array_id])

    elif 'array_May6' in sys.argv:
        configs = [
            {'priority_pred': 'max_desc_r',
             'return_type': 'max_desc_r',
             'num_molecules': int(10e6),
             'prune_at': prune_at,
             'actor_temperature': actor_temperature,
             'score_temperature': score_temperature,
             'num_rollout_actors': 2,
             'num_docking_threads': 4,
             'save_path': '/home/bengioem/scratch/lz'}

            for prune_at in [int(150e3), int(250e3)]
            for actor_temperature in [1, 5]
            for score_temperature in [1.5, 5]
            for run in [0]
        ]
        array_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        guided_search(configs[array_id])
    elif 1:

        guided_search({'priority_pred': 'max_desc_r', #'greedy_q',
                       'update_prio_on_refresh': True,
                       'return_type':'max_desc_r',
                       'save_path': os.path.join(summaries_dir, 'persistent_search_tree'),
                       'num_molecules': int(10e6),
                       'prune_at': int(200e3),
                       'num_rollout_actors': 1,
                       'num_docking_threads': 1,
                       'score_temperature': 1.5})
