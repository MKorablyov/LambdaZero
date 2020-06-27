#!/bin/bash
#SBATCH --account=rrg-bengioy-ad         # Yoshua pays for your job
#SBATCH --cpus-per-task=10               # Ask for 10 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=64G                        # Ask for 64 GB of RAM
#SBATCH --time=9:00:00                  # The job will run for 24 hours
#SBATCH -o /home/mkkr/scratch/slurm-%j.out  # Write the log in $SCRATCH

#Change Maksym's output directory, environment initialization, 
#python path, directory to the project, and other things to work.

source /home/mkkr/anaconda3/bin/activate th
export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects/"
cd /scratch/mkkr/Projects/LambdaZero/examples
python train_molecule.py $1
