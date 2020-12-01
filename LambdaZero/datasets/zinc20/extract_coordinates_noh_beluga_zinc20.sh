#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --array=0-999%100
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1:00:00
#SBATCH -o /home/mkkr/scratch/docking/log_extract_coordinates/%A_%a.out

module load python/3.6
module load nixpkgs/16.09
module load gcc/8.3.0 rdkit/2019.03.4
source /lustre03/project/6004852/mkkr/anaconda3/etc/profile.d/conda.sh
conda activate docking

cd $SLURM_TMPDIR
mkdir score_subsets
scp /home/mkkr/scratch/docking/score_subsets/subset_${SLURM_ARRAY_TASK_ID}.csv $SLURM_TMPDIR/score_subsets/

tar -xzf sdf_${SLURM_ARRAY_TASK_ID}.tar.gz
tar -xzf pdbqt_${SLURM_ARRAY_TASK_ID}.tar.gz
tar -xzf docked_${SLURM_ARRAY_TASK_ID}.tar.gz

export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/docking/LambdaZeroDev/"

cd /home/mkkr/scratch/docking/LambdaZeroDev/LambdaZero/datasets/zinc20/
python3 extract_coordinates_noh.py $SLURM_ARRAY_TASK_ID

scp ${SLURM_ARRAY_TASK_ID}.pth /home/mkkr/scratch/docking/graphs
