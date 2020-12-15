#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12                # Number of CPUs needed
#SBATCH --gres=gpu:4                      # Number of GPUs needed
#SBATCH --mem=64G                         # Amount of RAM needed
#SBATCH --time=48:00:00                   # The time for which the job runs
#SBATCH -o /scratch/<username>/slurm-%j.out  # Write the log in $SCRATCH


module load python/3.6 cuda/10.1 cudnn/7.6.5
module load nixpkgs/16.09  
module load gcc/7.3.0 
module load rdkit/2019.03.4
module load openbabel/2.4.1

export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_PATH/lib64
export PATH=/lustre03/project/6004852/mkkr/Programs/mgltools_x86_64Linux2_1.5.6/bin${PATH:+:${PATH}}

source /lustre03/project/6004852/mkkr/anaconda3/etc/profile.d/conda.sh
conda activate lz
export PYTHONPATH="${PYTHONPATH}:<path to your working directory>/LambdaZero" 

# cd to the experiment you would like to run and call the script. For eg.
cd ~/LambdaZero/LambdaZero/examples/bayesian_models/rl
python3 train_bayesian_reward.py

