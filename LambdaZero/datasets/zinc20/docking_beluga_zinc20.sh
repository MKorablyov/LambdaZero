#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=4:00:00
#SBATCH -o /home/mkkr/scratch/docking/log/%A_%a.out

module load python/3.6
module load nixpkgs/16.09
module load gcc/8.3.0 rdkit/2019.03.4
module load gcc/7.3.0
module load openbabel/2.4.1
source /lustre03/project/6004852/mkkr/anaconda3/etc/profile.d/conda.sh
conda activate docking

cd $SLURM_TMPDIR
mkdir Programs
mkdir Programs/vina
mkdir Programs/vina/bin
mkdir Datasets
mkdir Datasets/zinc20
scp /home/mkkr/projects/rrg-bengioy-ad/mkkr/Programs/vina/bin/vina Programs/vina/bin/
scp /home/mkkr/projects/rrg-bengioy-ad/mkkr/Programs/mgltools_x86_64Linux2_1.5.6.tar.gz Programs/
scp /home/mkkr/projects/rrg-bengioy-ad/lsky/Datasets/seh.tar.gz Datasets/

cd $SLURM_TMPDIR/Datasets
tar -xzf seh.tar.gz

cd $SLURM_TMPDIR/Programs
tar -xzf mgltools_x86_64Linux2_1.5.6.tar.gz

cd $SLURM_TMPDIR/Programs/mgltools_x86_64Linux2_1.5.6
./install.sh > /dev/null

export PATH=$SLURM_TMPDIR/Programs/mgltools_x86_64Linux2_1.5.6/bin:$PATH
export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/docking/LambdaZeroDev/"

cd /home/mkkr/scratch/docking/LambdaZeroDev/LambdaZero/datasets/zinc20/
python3 docking_beluga.py 10 $SLURM_ARRAY_TASK_ID

cd $SLURM_TMPDIR/Datasets/zinc20
tar -czf sdf_%a.tar.gz sdf
tar -czf mol2_%a.tar.gz mol2
tar -czf pdbqt_%a.tar.gz pdbqt
tar -czf docked_%a.tar.gz docked

scp sdf_%a.tar.gz /home/mkkr/scratch/docking/sdf_subsets
scp mol2_%a.tar.gz /home/mkkr/scratch/docking/mol2_subsets
scp pdbqt_%a.tar.gz /home/mkkr/scratch/docking/pdbqt_subsets
scp docked_%a.tar.gz /home/mkkr/scratch/docking/docked_subsets
scp subset_%a.csv /home/mkkr/scratch/docking/score_subsets
