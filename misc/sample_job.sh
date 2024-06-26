#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=40                # Number of CPUs needed
#SBATCH --gres=gpu:4                      # Number of GPUs needed
#SBATCH --mem=128G                         # Amount of RAM needed
#SBATCH --time=24:00:00                   # The time for which the job runs
#SBATCH -o /scratch/mkkr/slurm-%j.out  # Write the log in $SCRATCH

module load StdEnv/2018.3
module load python/3.6 cuda/10.1 cudnn/7.6.5
module load nixpkgs/16.09
module load gcc/7.3.0
module load rdkit/2019.03.4
module load openbabel/2.4.1

export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}
export PATH=/lustre03/project/6004852/mkkr/Programs/mgltools_x86_64Linux2_1.5.6/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_PATH/lib64
export PATH=/lustre03/project/6004852/mkkr/Programs/mgltools_x86_64Linux2_1.5.6/bin${PATH:+:${PATH}}

source /lustre03/project/6004852/mkkr/anaconda3/etc/profile.d/conda.sh
conda activate lz

# PYTHONPATH to the modified version of the repository (change both next lines)
export PYTHONPATH="${PYTHONPATH}:/lustre03/project/6004852/mkkr/Projects/7_LambdaZero"
cd /lustre03/project/6004852/mkkr/Projects/7_LambdaZero/LambdaZero/contrib
python train_rlbo.py rlbo_001

