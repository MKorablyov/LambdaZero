#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=12
#SBATCH --mem=12G
#SBATCH --time=4:00:00
#SBATCH -o /home/lsky/scratch/docking/%j.out

module load python/3.6
module load nixpkgs/16.09
module load gcc/8.3.0 rdkit/2019.03.4
module load gcc/7.3.0
module load openbabel/2.4.1
source /lustre03/project/6004852/mkkr/anaconda3/etc/profile.d/conda.sh
conda activate lz

cd $SLURM_TMPDIR
mkdir Programs
mkdir Programs/vina
mkdir Programs/vina/bin
mkdir Datasets
scp /home/mkkr/projects/rrg-bengioy-ad/mkkr/Programs/vina/bin/vina Programs/vina/bin/
scp /home/mkkr/projects/rrg-bengioy-ad/mkkr/Programs/mgltools_x86_64Linux2_1.5.6.tar.gz Programs/
scp /home/lsky/projects/rrg-bengioy-ad/lsky/Datasets/seh.tar.gz Datasets/

cd $SLURM_TMPDIR/Datasets
tar -xzf seh.tar.gz

cd $SLURM_TMPDIR/Programs
tar -xzf mgltools_x86_64Linux2_1.5.6.tar.gz

cd $SLURM_TMPDIR/Programs/mgltools_x86_64Linux2_1.5.6
./install.sh > /dev/null

export PATH=$SLURM_TMPDIR/Programs/mgltools_x86_64Linux2_1.5.6/bin:$PATH
export PYTHONPATH="${PYTHONPATH}:/home/lsky/scratch/LambdaZeroDev/"

cd /home/lsky/scratch/LambdaZeroDev/LambdaZero/datasets/postera/
python3 docking_beluga.py

scp $SLURM_TMPDIR/roc_auc.txt /home/lsky/projects/rrg-bengioy-ad/lsky/Datasets