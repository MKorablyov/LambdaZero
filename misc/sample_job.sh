#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12                # Number of CPUs needed
#SBATCH --gres=gpu:1                      # Number of GPUs needed
#SBATCH --mem=64G                         # Amount of RAM needed
#SBATCH --time=12:00:00                   # The time for which the job runs
#SBATCH -o /scratch/mkkr/slurm-%j.out  # Write the log in $SCRATCH


module load python/3.6 cuda/10.1 cudnn/7.6.5
module load nixpkgs/16.09
module load gcc/7.3.0
module load rdkit/2019.03.4
module load openbabel/2.4.1
export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}
export PATH=/lustre03/project/6004852/mkkr/Programs/mgltools_x86_64Linux2_1.5.6/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_PATH/lib64
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_PATH/lib64

source /lustre03/project/6004852/mkkr/anaconda3/etc/profile.d/conda.sh
conda activate lz


export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects/15_LambdaZero"
cd /home/mkkr/scratch/Projects/15_LambdaZero/LambdaZero/examples/mpnn
python train_mpnn.py
