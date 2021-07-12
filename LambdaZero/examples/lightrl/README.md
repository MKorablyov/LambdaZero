### Requirements
- `liftoff` (`pip install git+git://github.com/tudor-berariu/liftoff.git#egg=liftoff`)

### Scripts

- train ppo: `train_light.py`
- PPO Algorithm: `...lightrl/a2c_ppo_acktr/algo/ppo.py` or `...lightrl/a2c_ppo_acktr/algo/ppo_two_v.py` 
- PPO rollout storage: `...lightrl/utils/storage.py` or `...lightrl/utils/storage_two_v.py` 
- Environment wrappers `lightrl/env/gym_wrappers.py` - Wrappers should be used for e.g. for generating different rewards
- Models `...lightrl/models`
- Configs: `...lightrl/configs` 

### Some config customizations:

e.g. Two value head:
2nd reward is calculated based on `extra_r` value in `info` dict returned by env. Thus, we should have a wrapper that populates this value (e.g. `SeenLinearInfo` or `SeenInfo`)

```
env_cfg:
  wrapper: ["SeenLinearInfo", "ProxyCandidateReward"] # 
model:
  num_out: 3  # 3 scalars 1 for stop - 1 for each value head
algo: "ppo_twov"
```


### Usage 

Run 1 config (Best config Jul/2021): 

`liftoff train_light.py LambdaZero/examples/lightrl/configs/ppo_best_jul21.yaml`

Prepare & run batch of configs (example):
1. `liftoff-prepare LambdaZero/examples/lightrl/configs/rl_grid/ --runs-no 3 --results-path /scratch/andrein/lz2/results/  --do`
 Prepare all experiments from folder `rl_grid` (based on the folder's `default.yaml` and `config*.yaml` files) - for `runs-no` seeds / config (3 in our example) - and generate experiment folders at the specified `results-path` (should be on scratch).
   
example out:

``` 
...
Summary:
	Sub-experiments: 24 | New: 24 | Existing: 0
	Runs: 72 | New: 72 | Existing: 0 | Written: 72

Experiment configured in /scratch/andrein/lz2/results/2021Jul12-115643_rl_grid/
```

2. Run array of jobs with e.g. by specifing array number (`1-72`) and `rel_path` for the experiments folder (generated above at step 1.). Each job will start and run `liftoff` with 1 config that has not been processed yet:

* TODO adjust to your own python env & paths (e.g. scratch) 
```
#!/bin/bash
#SBATCH --array=1-72
#SBATCH --account=rrg-bengioy-ad #def-dprecup  #def-bengioy  # rrg-bengioy-ad
#SBATCH -o /scratch/andrein/Lambda/slurm-%j.out
#SBATCH --cpus-per-task=18                   # Ask for 2 CPUs
#SBATCH --gres=gpu:1                          # Ask for 1 GPU
#SBATCH --mem=44Gb                             # Ask for 10 GB of RAM
#SBATCH --time=12:00:00                        # The job will run for 3 hours

source $HOME/.bashrc

module load StdEnv/2018.3
module load python/3.7 cuda/10.1 cudnn/7.6.5
module load nixpkgs/16.09
module load gcc/7.3.0
module load rdkit/2019.03.4
module load openbabel/2.4.1
export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}
#export PATH=/lustre03/project/6004852/mkkr/Programs/mgltools_x86_64Linux2_1.5.6/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_PATH/lib64
#export PATH=/lustre03/project/6004852/mkkr/Programs/mgltools_x86_64Linux2_1.5.6/bin${PATH:+:${PATH}}
module load python/3.7 cuda/10.1 cudnn/7.6.5
module load httpproxy

source /home/andrein/miniconda/etc/profile.d/conda.sh
conda activate alz7
source deactivate
source activate alz7
conda activate alz7

# PYTHONPATH to the modified version of the repository (change both next lines)
export PYTHONPATH="${PYTHONPATH}:/home/andrein/lzbranch2"


WANDB_DISABLE_CODE=true

echo "Runnig sbatch array job $SLURM_ARRAY_TASK_ID"
echo "Running sbatch job $SLURM_ARRAY_JOB_ID"

rel_path="/scratch/andrein/lz2/results/2021Jul12-115643_rl_grid/"

liftoff train_light.py --max-runs 1 --no-detach ${rel_path}


```
