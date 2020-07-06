#!/bin/bash
#SBATCH --account=rrg-bengioy-ad               # Yoshua pays for your job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G                               # Ask for 8 GB of RAM
#SBATCH --time=96:00:00                        # The job will run for 24 hours
#SBATCH --array=1-200                          # 100 of this
#SBATCH -o /home/mkkr/scratch/slurm-%j.out  # Write the log in $SCRATCH

#Change Maksym's directories to match your own directories

source /home/mkkr/anaconda3/bin/activate th
export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects"

cd /home/mkkr/scratch/Projects/LambdaZero/datasets
python actor_dock.py
