import toy_1d_seq as toy
import ray
import copy
from itertools import count
import torch
import tqdm
def array_mar_30():
    cnt = count(0)
    args = [
        {'learning_rate': lr,
         'horizon': horizon,
         'n_hid': nh,
         'learning_method': method,
         'save_path': f'results/mar_24/{next(cnt)}.pkl.gz',
         'uniform_sample_prob': alpha,
         'n_train_steps': 8000,
         }
        for lr in [1e-4, 2e-4, 5e-4]
        for horizon in range(8,9)
        for nh in [64, 128, 256]
        for alpha in [0.001, 0.01, 0.05]
        for method in ['td']]
    return args
torch.set_num_threads(2)
toy.parser.add_argument('--array', default='mar_30')
args = toy.parser.parse_args()
all_args = []
for i in eval('array_'+args.array)():
    a = copy.copy(args)
    for k,v in i.items():
        setattr(a, k, v)
    all_args.append(a)
ray.init(num_cpus=8)
rmain = ray.remote(toy.main)
jobs = [rmain.remote(i) for i in all_args]
print(len(jobs), 'jobs')
with tqdm.tqdm(total=len(jobs), smoothing=0) as t:
    while len(jobs):
        ready, jobs = ray.wait(jobs)
        for i in ready:
            t.update()