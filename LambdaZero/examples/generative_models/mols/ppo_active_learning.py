import time, os, sys, pickle, gzip, argparse, os.path as osp
import torch
import concurrent.futures
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import LambdaZero.utils
from main_flow import make_model
from rdkit import Chem
from rdkit import DataStructs
from mol_active_learning_v2 import ProxyDataset, Proxy, Docker
from ppo import PPODataset as GenModelDataset

datasets_dir, programs_dir, summaries_dir = LambdaZero.utils.get_external_dirs()
if 'SLURM_TMPDIR' in os.environ:
    #     print("Syncing locally")
    tmp_dir = os.environ['SLURM_TMPDIR'] + '/lztmp/'
else:
    tmp_dir = osp.join(datasets_dir, "temp_docking")
os.makedirs(tmp_dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--proxy_learning_rate", default=2.5e-4, help="Learning rate", type=float)
parser.add_argument("--proxy_dropout", default=0.1, help="MC Dropout in Proxy", type=float)
parser.add_argument("--proxy_weight_decay", default=1e-5, help="Weight Decay in Proxy", type=float)
parser.add_argument("--proxy_mbsize", default=64, help="Minibatch size", type=int)
parser.add_argument("--proxy_opt_beta", default=0.9, type=float)
parser.add_argument("--proxy_nemb", default=64, help="#hidden", type=int)
parser.add_argument("--proxy_num_iterations", default=25000, type=int) # fixme [maksym] 25000
parser.add_argument("--num_init_examples", default=1024, type=int) # fixme [maksym] 1024
parser.add_argument("--num_outer_loop_iters", default=25, type=int)
parser.add_argument("--num_samples", default=256, type=int)
parser.add_argument("--proxy_num_conv_steps", default=6, type=int) # [ maksym] made 6 from 12
parser.add_argument("--proxy_repr_type", default='atom_graph')
parser.add_argument("--proxy_model_version", default='v2')
parser.add_argument("--save_path", default='results/')
parser.add_argument("--cpu_req", default=8)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--include_nblocks", action='store_true')

# gen_model
parser.add_argument("--learning_rate", default=5e-4, help="Learning rate", type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--opt_beta", default=0.9, type=float)
parser.add_argument("--opt_beta2", default=0.99, type=float)
parser.add_argument("--opt_epsilon", default=1e-8, type=float)
parser.add_argument("--kappa", default=0.1, type=float)
parser.add_argument("--nemb", default=256, help="#hidden", type=int)
parser.add_argument("--min_blocks", default=2, type=int)
parser.add_argument("--max_blocks", default=6, type=int)
parser.add_argument("--num_iterations", default=1000, type=int) # fixme [maksym] 4000
parser.add_argument("--num_conv_steps", default=6, type=int) # maksym: do 6
parser.add_argument("--log_reg_c", default=1e-2, type=float)
parser.add_argument("--reward_exp", default=4, type=float)
parser.add_argument("--reward_norm", default=8, type=float)
parser.add_argument("--R_min", default=0.1, type=float)
parser.add_argument("--sample_prob", default=1, type=float)
parser.add_argument("--clip_grad", default=5.0, type=float)
parser.add_argument("--clip_loss", default=0, type=float)
parser.add_argument("--random_action_prob", default=0.05, type=float)
parser.add_argument("--leaf_coef", default=10, type=float)
parser.add_argument("--replay_mode", default='online', type=str)
parser.add_argument("--bootstrap_tau", default=0, type=float)
parser.add_argument("--weight_decay", default=0, type=float)
parser.add_argument("--array", default='')
parser.add_argument("--repr_type", default='block_graph')
parser.add_argument("--model_version", default='v4')
parser.add_argument("--run", default=0, help="run", type=int)
# parser.add_argument("--proxy_path", default='results/proxy__6/')
parser.add_argument("--balanced_loss", default=True)
parser.add_argument("--floatX", default='float64')
parser.add_argument("--ppo_clip", default=0.2, type=float)
parser.add_argument("--ppo_entropy_coef", default=1e-4, type=float)
parser.add_argument("--ppo_num_samples_per_step", default=256, type=float)
parser.add_argument("--ppo_num_epochs_per_step", default=32, type=float)


def train_generative_model(args, model, proxy, dataset, num_steps=None, do_save=True):
    debug_no_threads = False
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1

    tau = args.bootstrap_tau
    if args.bootstrap_tau > 0:
        target_model = deepcopy(model)

    if do_save:
        exp_dir = f'{args.save_path}/{args.array}_{args.run}/'
        os.makedirs(exp_dir, exist_ok=True)

    model = model.double()
    proxy.proxy = proxy.proxy.double()

    print("type dataset", type(dataset))
    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)



    def save_stuff():
        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/train_info.pkl.gz', 'wb'))


    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2))
    #opt = torch.optim.SGD(model.parameters(), args.learning_rate)

    #tf = lambda x: torch.tensor(x, device=device).float()
    tf = lambda x: torch.tensor(x, device=device).to(args.floatX)
    tint = lambda x: torch.tensor(x, device=device).long()

    mbsize = args.mbsize
    ar = torch.arange(mbsize)

    last_losses = []

    def stop_everything():
        print('joining')
        dataset.stop_samplers_and_join()
    #_stop[0] = stop_everything

    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()

    #loginf = 1000 # to prevent nans
    #log_reg_c = args.log_reg_c
    #clip_loss = tf([args.clip_loss])
    clip_param = args.ppo_clip
    entropy_coef = args.ppo_entropy_coef

    for i in range(num_steps):
        samples = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(dataset._get_sample_model)
                       for i in range(args.ppo_num_samples_per_step)]
            for future in tqdm(concurrent.futures.as_completed(futures), leave=False):
                samples += future.result()
        for j in range(args.ppo_num_epochs_per_step):
            idxs = dataset.train_rng.randint(0, len(samples), args.mbsize)
            mb = [samples[i] for i in idxs]

            s, a, r, d, lp, v, G, A = dataset.sample2batch(zip(*mb))

            s_o, m_o = model(s)
            new_logprob = -model.action_negloglikelihood(s, a, 0, s_o, m_o)
            values = m_o[:, 1]
            ratio = torch.exp(new_logprob - lp)

            surr1 = ratio * A
            surr2 = torch.clamp(ratio, 1.0 - clip_param,
                                1.0 + clip_param) * A
            action_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (G - values).pow(2).mean()
            m_p, s_p = model.out_to_policy(s, s_o, m_o)
            p = torch.zeros_like(m_p).index_add_(0, s.stems_batch, (s_p * torch.log(s_p)).sum(1))
            p = p + m_p * torch.log(m_p)
            entropy = -p.mean()
            loss = action_loss + value_loss - entropy * entropy_coef

            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(),
                                                args.clip_grad)
            opt.step()

            last_losses.append((loss.item(), value_loss.item(), entropy.item()))
            train_losses.append((loss.item(), value_loss.item(), entropy.item()))

        if not i % 10:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            print(i, last_losses, G.mean().item())
            print('time:', time.time() - time_last_check)
            time_last_check = time.time()
            last_losses = []

        if not i % 25 and do_save:
            save_stuff()

    stop_everything()
    if do_save:
        save_stuff()
    return model, dataset, {'train_losses': train_losses,
                            'test_losses': test_losses,
                            'test_infos': test_infos,
                            'train_infos': train_infos}


def sample_and_update_dataset(args, model, proxy_dataset, generator_dataset, docker):

    sampled_mols = []
    for i in range(args.num_samples):
        traj = generator_dataset._get_sample_model()

        mol = traj[-1][0]
        dockscore = docker.eval(mol, norm=False)
        mol.reward = proxy_dataset.r2r(dockscore=dockscore)
        # mol.smiles = s
        sampled_mols.append(mol)

    print("Computing distances")
    dists = []
    for m1, m2 in zip(sampled_mols, sampled_mols[1:] + sampled_mols[:1]):
        dist = DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m1.mol), Chem.RDKFingerprint(m2.mol))
        dists.append(dist)
    print("Get batch rewards")
    rewards = []
    for m in sampled_mols:
        rewards.append(m.reward)
    print("Add to dataset")
    proxy_dataset.add_samples(sampled_mols)
    return proxy_dataset, {
        'dists': dists, 'rewards': rewards, 'reward_mean': np.mean(rewards), 'reward_max': np.max(rewards),
        'dists_mean': np.mean(dists), 'dists_sum': np.sum(dists)
    }



def main(args):

    original_args = args
    bpath = osp.join(datasets_dir, "fragdb/blocks_PDB_105.json")
    device = torch.device('cuda')
    proxy_repr_type = args.proxy_repr_type
    repr_type = args.repr_type

    docker = Docker(tmp_dir, cpu_req=args.cpu_req)
    args.repr_type = proxy_repr_type
    args.replay_mode = "dataset"
    proxy_dataset = ProxyDataset(args, bpath, device, floatX=torch.float)
    proxy_dataset.load_h5(osp.join(datasets_dir, "brutal_dock/seh",
                                   "dock_db_1619111711tp_2021_04_22_13h.h5"),
                          args, num_examples=args.num_init_examples)

    exp_dir = f'{args.save_path}/ppo_proxy_{args.array}_{args.run}/'
    os.makedirs(exp_dir, exist_ok=True)

    print(len(proxy_dataset.train_mols), 'train mols')
    print(len(proxy_dataset.test_mols), 'test mols')
    print(args)

    proxy = Proxy(args, bpath, device)
    train_metrics = []
    metrics = []
    proxy.train(proxy_dataset)

    # fixme [ maksym] in Moksh's version GM was retrained all the time; I keep updating same model here
    # maybe it would be rational to add some noise to the weights
    args.sample_prob = 1
    args.repr_type = repr_type
    args.replay_mode = "online"

    gen_model_dataset = GenModelDataset(args, bpath, device)
    print("0 type gen model dataset", type(gen_model_dataset))
    model = make_model(args, gen_model_dataset.mdp, out_per_mol=2)
    if args.floatX == 'float64':
        model = model.double()

    model.to(device)

    for i in range(args.num_outer_loop_iters):
        # Initialize model and dataset for training generator
        args.sample_prob = 1
        args.repr_type = repr_type
        args.replay_mode = "online"
        gen_model_dataset = GenModelDataset(args, bpath, device)

        # train generative model with with proxy
        print(f"Training model: {i}")
        model, gen_model_dataset, training_metrics = train_generative_model(original_args, model, proxy,
                                                                            gen_model_dataset, do_save=False)

        print(f"Sampling model: {i}")
        # sample molecule batch for generator and update dataset with docking scores for sampled batch
        _proxy_dataset, batch_metrics = sample_and_update_dataset(args, model, proxy_dataset, gen_model_dataset, docker)
        print(
            f"Batch Metrics: dists_mean: {batch_metrics['dists_mean']}, dists_sum: {batch_metrics['dists_sum']},"
            f" reward_mean: {batch_metrics['reward_mean']}, reward_max: {batch_metrics['reward_max']}")

        args.sample_prob = 0
        args.repr_type = proxy_repr_type
        args.replay_mode = "dataset"
        train_metrics.append(training_metrics)
        metrics.append(batch_metrics)

        proxy_dataset = ProxyDataset(args, bpath, device, floatX=torch.float)
        proxy_dataset.train_mols.extend(_proxy_dataset.train_mols)
        proxy_dataset.test_mols.extend(_proxy_dataset.test_mols)

        proxy = Proxy(args, bpath, device)


        pickle.dump({'train_metrics': train_metrics,
                     'batch_metrics': metrics,
                     'all_mols': [mol.smiles for mol in proxy_dataset.train_mols],
                     'args': args},
                    gzip.open(f'{exp_dir}/info.pkl.gz', 'wb'))

        print(f"Updating proxy: {i}")
        # update proxy with new data
        proxy.train(proxy_dataset)



if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()

    #print("parsed args")
    #if len(sys.argv) >= 2:
    #    run = sys.argv[1]
    #    #print("config idx", config_idx)
    #else:
    #    run = 0

    run = 5
    configs = [{"run": 0},
               {"run": 1,"num_init_examples": 512, "num_samples": 256},
               {"run": 2, "num_init_examples": 2048, "num_samples": 256},
               {"run": 3, "num_init_examples": 8192, "num_samples": 256},

               {"run": 4, "reward_exp": 4},
               {"run": 5, "num_init_examples": 512, "num_samples": 256, "reward_exp": 4},
               {"run": 6, "num_init_examples": 2048, "num_samples": 256, "reward_exp": 4},
               {"run": 7, "num_init_examples": 8192, "num_samples": 256, "reward_exp": 4},

               {"run": 8, "reward_exp": 2},
               {"run": 9, "num_init_examples": 512, "num_samples": 256, "reward_exp": 2},
               {"run": 10, "num_init_examples": 2048, "num_samples": 256, "reward_exp": 2},
               {"run": 11, "num_init_examples": 8192, "num_samples": 256, "reward_exp": 2},
               ]

    for k, v in configs[args.run].items(): setattr(args, k, v)
    print("args",args)
    main(args)