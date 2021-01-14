# source: docs.mila.quebec
# case 1: to get an interactive allocation run: salloc --time=1:0:0 --cpus-per-task=4 --gres=gpu:1 --mem=32G --account=rrg-bengioy-ad
# case 2: to submit a beluga batch job don't run this scrip directly, instead modify and run beluga_gpu.sh or beluga_gpu.sh in this folder

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


# in order to run LambdaZero, you may also want to add the particular version you have to PYTHONPATH
# export PYTHONPATH="${PYTHONPATH}:/home/mkkr/scratch/Projects/15_LambdaZero"